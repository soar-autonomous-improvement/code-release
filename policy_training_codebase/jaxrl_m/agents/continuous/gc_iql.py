import copy
from functools import partial
from typing import *

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import distrax
from flax.core import FrozenDict

from jaxrl_m.agents.continuous.iql import (
    expectile_loss,
    iql_actor_loss,
    iql_critic_loss,
    iql_value_loss,
)
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import GCEncodingWrapper
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import Batch, Data, Params, PRNGKey
from jaxrl_m.networks.actor_critic_nets import Critic, Policy, ValueCritic, ensemblize
from jaxrl_m.networks.mlp import MLP


class GCIQLAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        batch_size = batch["terminals"].shape[0]
        neg_goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)

        # selects a portion of goals to make negative
        def get_goals_rewards_masks_mc_returns(key):
            neg_goal_mask = (
                jax.random.uniform(key, (batch_size,))
                < self.config["negative_proportion"]
            )
            goal_indices = jnp.where(
                neg_goal_mask, neg_goal_indices, jnp.arange(batch_size)
            )
            new_goals = jax.tree_map(lambda x: x[goal_indices], batch["goals"])
            new_rewards = jnp.where(neg_goal_mask, -1, batch["rewards"])
            new_masks = jnp.where(neg_goal_mask, 1, batch["masks"])
            new_mc_returns = jnp.where(
                neg_goal_mask, -1 / (1 - self.config["discount"]), batch["mc_returns"]
            )
            return new_goals, new_rewards, new_masks, new_mc_returns

        def critic_loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            goals, rewards, masks, mc_returns = get_goals_rewards_masks_mc_returns(key)

            rng, key = jax.random.split(rng)
            next_v = self.forward_target_value((batch["next_observations"], goals), key)
            target_q = rewards + self.config["discount"] * next_v * masks
            rng, key = jax.random.split(rng)
            q = self.forward_critic(
                (batch["observations"], goals),
                batch["actions"],
                key,
                grad_params=params,
            )
            loss, info = iql_critic_loss(q, target_q)
            info["mc_error"] = jnp.square(q - mc_returns).mean()
            info["mc_return_mean"] = mc_returns.mean()
            info["mc_return_max"] = mc_returns.max()
            info["mc_return_min"] = mc_returns.min()
            return loss, info

        def value_loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            goals, _, _, _ = get_goals_rewards_masks_mc_returns(key)

            rng, key = jax.random.split(rng)
            q = self.forward_target_critic(
                (batch["observations"], goals), batch["actions"], key
            )  # no gradient
            q = jnp.min(q, axis=0)  # min over 2 Q functions
            rng, key = jax.random.split(rng)
            v = self.forward_value(
                (batch["observations"], goals), key, grad_params=params
            )

            return iql_value_loss(q, v, self.config["expectile"])

        def actor_loss_fn(params, rng):            
            rng, key = jax.random.split(rng)
            
            if self.config["update_actor_with_target_adv"]:
                critic_fn = self.forward_target_critic
            else:
                # Seohong: not using the target will make updates faster
                critic_fn = self.forward_critic
            q = critic_fn(
                (batch["observations"], batch["goals"]), batch["actions"], key
            )  # no gradient
            q = jnp.min(q, axis=0)  # min over 2 Q functions

            rng, key = jax.random.split(rng)
            v = self.forward_value(
                (batch["observations"], batch["goals"]), key
            )  # no gradients

            rng, key = jax.random.split(rng)
            dist = self.forward_policy(
                (batch["observations"], batch["goals"]), key, grad_params=params
            )

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
        new_state, info, grad_norm = self.state.apply_loss_fns(
            loss_fns,
            pmap_axis=pmap_axis,
            has_aux=True,
            return_grad_norm=True,
        )
        info["grad_norm"] = grad_norm

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # Log learning rates
        for name, opt_state in new_state.opt_states.items():
            if (
                hasattr(opt_state, "hyperparams")
                and "learning_rate" in opt_state.hyperparams.keys()
            ):
                info[f"{name}_lr"] = opt_state.hyperparams["learning_rate"]

        return self.replace(state=new_state), info
    
    def forward_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
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
        # forward the q function on 1 action on each state
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_policy(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
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

    def forward_value(
        self,
        observations: Union[Data, Tuple[Data, Data]],
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
    
    def forward_target_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )
    
    def forward_target_value(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        rng: Optional[PRNGKey] = None,
    ) -> jax.Array:
        """
        Forward pass for target value network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_value(
            observations, rng=rng, grad_params=self.state.target_params
        )

    def forward_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
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
        # forward the q function on 1 action on each state
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            observations,
            actions,
            name="critic",
            rngs={"dropout": rng} if train else {},
            train=train,
        )

    def forward_policy(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        rng: Optional[PRNGKey] = None,
        *,
        grad_params: Optional[Params] = None,
        train: bool = True,
    ) -> distrax.Distribution:
        """
        Forward pass for policy network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
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

    def forward_value(
        self,
        observations: Union[Data, Tuple[Data, Data]],
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

    def forward_target_critic(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        actions: jax.Array,
        rng: Optional[PRNGKey] = None,
    ) -> jax.Array:
        """
        Forward pass for target critic network.
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.forward_critic(
            observations, actions, rng=rng, grad_params=self.state.target_params
        )

    def forward_target_value(
        self,
        observations: Union[Data, Tuple[Data, Data]],
        rng: Optional[PRNGKey] = None,
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
        goals: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax=False,
    ) -> jnp.ndarray:
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals),
            temperature=temperature,
            name="actor",
        )
        if argmax:
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions, dist.mode()

    @jax.jit
    def get_debug_metrics(self, batch, gripper_close_val=None, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        v = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            name="value",
        )
        next_v = self.state.apply_fn(
            {"params": self.state.target_params},
            (batch["next_observations"], batch["goals"]),
            name="value",
        )
        target_q = batch["rewards"] + self.config["discount"] * next_v * batch["masks"]
        q = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            batch["actions"],
            name="critic",
        )

        metrics = {
            "log_probs": log_probs,
            "mse": mse,
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
                (batch["observations"], batch["goals"]),
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
        goals: FrozenDict,
        actions: jnp.ndarray,
        # Model architecture
        encoder_def: nn.Module,
        shared_encoder: bool = True,
        shared_goal_encoder: bool = True,
        early_goal_concat: bool = False,
        use_proprio: bool = False,
        negative_proportion: float = 0.0,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "dropout": 0.0,
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
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
        dropout_target_networks=True,
        update_actor_with_target_adv=True,
    ):
        if early_goal_concat:
            # passing None as the goal encoder causes early goal concat
            goal_encoder_def = None
        else:
            if shared_goal_encoder:
                goal_encoder_def = encoder_def
            else:
                goal_encoder_def = copy.deepcopy(encoder_def)

        if encoder_def is not None:
            encoder_def = GCEncodingWrapper(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
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
            # shared_goal_encoder, but I haven't tested it.
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

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            actor=[(observations, goals)],
            value=[(observations, goals)],
            critic=[(observations, goals), actions],
        )["params"]

        # create optimizers
        actor_optimizer, actor_lr_schedule = make_optimizer(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            cosine_decay_steps=actor_decay_steps if actor_decay_steps is not None else None,
            weight_decay=0.001,
            beta2=0.98,
            clip_grad_norm=1.0,
            return_lr_schedule=True,
        )
        value_optimizer, value_lr_schedule = make_optimizer(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            cosine_decay_steps=None,
            weight_decay=0.001,
            beta2=0.98,
            clip_grad_norm=1.0,
            return_lr_schedule=True,
        )
        critic_optimizer, critic_lr_schedule = make_optimizer(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            cosine_decay_steps=None,
            weight_decay=0.001,
            beta2=0.98,
            clip_grad_norm=1.0,
            return_lr_schedule=True,
        )
        txs = {
            "actor": actor_optimizer,
            "value": value_optimizer,
            "critic": critic_optimizer,
        }
        lr_schedules = {
            "actor": actor_lr_schedule,
            "value": value_lr_schedule,
            "critic": critic_lr_schedule,
        }

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
                dropout_target_networks=dropout_target_networks,
                negative_proportion=negative_proportion,
                update_actor_with_target_adv=update_actor_with_target_adv,
            )
        )
        return cls(state, config, lr_schedules)
