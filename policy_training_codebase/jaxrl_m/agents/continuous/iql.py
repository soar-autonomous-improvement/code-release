import copy
from functools import partial
from typing import Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core import FrozenDict

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.networks.actor_critic_nets import Critic, Policy, ValueCritic, ensemblize
from jaxrl_m.networks.mlp import MLP


def expectile_loss(diff, expectile=0.5):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


def iql_value_loss(q, v, expectile):
    value_loss = expectile_loss(q - v, expectile)
    return value_loss.mean(), {
        "value_loss": value_loss.mean(),
        "uncentered_loss": jnp.mean((q - v) ** 2),
        "v": v.mean(),
    }


def iql_critic_loss(q, q_target):
    critic_loss = jnp.square(q - q_target)
    return critic_loss.mean(), {
        "td_loss": critic_loss.mean(),
        "q": q.mean(),
    }


def iql_actor_loss(q, v, dist, actions, temperature=1.0, adv_clip_max=100.0, mask=None):
    adv = q - v

    exp_adv = jnp.exp(adv / temperature)
    exp_adv = jnp.minimum(exp_adv, adv_clip_max)

    log_probs = dist.log_prob(actions)
    actor_loss = -(exp_adv * log_probs)

    if mask is not None:
        actor_loss *= mask
        actor_loss = jnp.sum(actor_loss) / jnp.sum(mask)
    else:
        actor_loss = jnp.mean(actor_loss)

    behavior_mse = jnp.square(dist.mode() - actions).sum(-1)

    return actor_loss, {
        "actor_loss": actor_loss,
        "behavior_logprob": log_probs.mean(),
        "behavior_mse": behavior_mse.mean(),
        "adv_mean": adv.mean(),
        "adv_std": adv.std(),
        "adv_max": adv.max(),
        "adv_min": adv.min(),
        "predicted actions": dist.mode(),
        "dataset actions": actions,
    }


class IQLAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        rng, new_rng = jax.random.split(self.state.rng)

        def critic_loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            next_v = self.forward_target_value(batch["next_observations"], key)
            target_q = (
                batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
            )
            rng, key = jax.random.split(rng)
            q = self.forward_critic(
                batch["observations"], batch["actions"], key, grad_params=params
            )
            return iql_critic_loss(q, target_q)

        def value_loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            q = self.forward_target_critic(
                batch["observations"], batch["actions"], key
            )  # no gradient
            q = jnp.min(q, axis=0)  # min over 2 Q functions

            rng, key = jax.random.split(rng)
            v = self.forward_value(batch["observations"], key, grad_params=params)

            # expectile loss
            return iql_value_loss(q, v, self.config["expectile"])

        def actor_loss_fn(params, rng):
            rng, key = jax.random.split(rng)

            if self.config["update_actor_with_target_adv"]:
                critic_fn = self.forward_target_critic
            else:
                # Seohong: not using the target will make updates faster
                critic_fn = self.forward_critic
            q = critic_fn(batch["observations"], batch["actions"], key)  # no gradient
            q = jnp.min(q, axis=0)  # min over 2 Q functions

            rng, key = jax.random.split(rng)
            v = self.forward_value(batch["observations"], key)  # no gradients

            rng, key = jax.random.split(rng)
            dist = self.forward_policy(batch["observations"], key, grad_params=params)
            mask = batch.get("actor_loss_mask", None)
            return iql_actor_loss(
                q,
                v,
                dist,
                batch["actions"],
                self.config["temperature"],
                mask=mask,
            )

        loss_fns = {
            "critic": critic_loss_fn,
            "value": value_loss_fn,
            "actor": actor_loss_fn,
        }

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # update rng
        new_state = new_state.replace(rng=new_rng)

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info

    def forward_policy(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ):
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients)
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="actor",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        qs = self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )
        return qs

    def forward_target_critic(
        self,
        observations: Data,
        actions: jax.Array,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_value(
        self,
        observations: Data,
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> jax.Array:
        """
        Forward pass for value network.
        Pass grad_params
        """
        if train:
            assert rng is not None, "Must specify rng when training"
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            name="value",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_target_value(
        self,
        observations: Data,
        rng: PRNGKey,
    ) -> jax.Array:
        """
        Forward pass for target value network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_value(
            observations, rng=rng, grad_params=self.state.target_params
        )
    
    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        argmax=False,
    ) -> jnp.ndarray:
        dist = self.forward_policy(observations, seed, train=False)
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, gripper_close_val=None, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        v = self.state.apply_fn(
            {"params": self.state.params}, batch["observations"], name="value"
        )
        next_v = self.state.apply_fn(
            {"params": self.state.target_params},
            batch["next_observations"],
            name="value",
        )
        target_q = batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
        q = self.state.apply_fn(
            {"params": self.state.params},
            batch["observations"],
            batch["actions"],
            name="critic",
        )

        metrics = {
            "log_probs": log_probs,
            "mse": ((dist.mode() - batch["actions"]) ** 2).sum(-1),
            "pi_actions": pi_actions,
            "online_v": v,
            "online_q": q,
            "target_q": target_q,
            "value_err": expectile_loss(target_q - v, self.config["expectile"]),
            "td_err": jnp.square(target_q - q),
            "advantage": target_q - v,
            "qf_advantage": q - v,
        }

        if gripper_close_val is not None:
            gripper_close_q = self.state.apply_fn(
                {"params": self.state.params},
                batch["observations"],
                jnp.broadcast_to(gripper_close_val, batch["actions"].shape),
                name="critic",
            )
            metrics.update(
                {
                    "gripper_close_q": gripper_close_q,
                    "gripper_close_adv": gripper_close_q - v,
                }
            )

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
            "dropout": 0.0,
        },
        # Optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        # Algorithm config
        discount=0.95,
        expectile=0.9,
        temperature=1.0,
        target_update_rate=0.002,
        update_actor_with_target_adv=True,
    ):
        if encoder_def is not None:
            encoder_def = EncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )

        if shared_encoder:
            encoders = {
                "actor": encoder_def,
                "value": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": encoder_def,
                "value": copy.deepcopy(encoder_def),
                "critic": copy.deepcopy(encoder_def),
            }

        network_kwargs["activate_final"] = True
        networks = {
            "actor": Policy(
                encoders["actor"],
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs,
            ),
            "value": ValueCritic(encoders["value"], MLP(**network_kwargs)),
            "critic": Critic(
                encoders["critic"],
                network=ensemblize(partial(MLP, **network_kwargs), 2)(
                    name="critic_ensemble"
                ),
            ),
        }

        model_def = ModuleDict(networks)

        # no decay
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=warmup_steps + 1,
            end_value=learning_rate,
        )
        lr_schedules = {
            "actor": lr_schedule,
            "value": lr_schedule,
            "critic": lr_schedule,
        }
        if actor_decay_steps is not None:
            lr_schedules["actor"] = optax.warmup_cosine_decay_schedule(
                init_value=0.0,
                peak_value=learning_rate,
                warmup_steps=warmup_steps,
                decay_steps=actor_decay_steps,
                end_value=0.0,
            )
        txs = {k: optax.adam(v) for k, v in lr_schedules.items()}

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=observations,
            value=observations,
            critic=[observations, actions],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                discount=discount,
                temperature=temperature,
                target_update_rate=target_update_rate,
                expectile=expectile,
                update_actor_with_target_adv=update_actor_with_target_adv,
            )
        )
        return cls(state, config)
