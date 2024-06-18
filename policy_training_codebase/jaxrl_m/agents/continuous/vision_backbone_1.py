import copy
from functools import partial
from typing import Any, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import GCEncodingWrapper, LCEncodingWrapper
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import Batch, PRNGKey
from jaxrl_m.networks.actor_critic_nets import Policy, ValueCritic
from jaxrl_m.networks.mlp import MLP


class VisionBackbone1(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    lr_schedule: Any = nonpytree_field()
    config: dict = nonpytree_field()

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, rng):
            rng, key = jax.random.split(rng)
            dist = self.state.apply_fn(
                {"params": params},
                (batch["observations"], batch["goals"]),
                temperature=1.0,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )
            pi_actions = dist.mode()
            log_probs = dist.log_prob(batch["actions"])
            mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)
            actor_loss = -(log_probs).mean()
            actor_std = dist.stddev().mean(axis=1)

            rng, key = jax.random.split(rng)
            mc_estimates = self.state.apply_fn(
                {"params": params},
                (batch["observations"], batch["goals"]),
                train=True,
                rngs={"dropout": key},
                name="mc_return_regressor",
            )
            mc_mse = ((mc_estimates - batch["mc_returns"]) ** 2).mean()

            loss = (
                self.config["gcbc_loss_ratio"] * actor_loss
                + (1 - self.config["gcbc_loss_ratio"]) * mc_mse
            )

            return loss, {
                "actor_loss": actor_loss,
                "actor_mse": mse.mean(),
                "actor_log_probs": log_probs.mean(),
                "actor_pi_actions": pi_actions.mean(),
                "actor_mean_std": actor_std.mean(),
                "actor_max_std": actor_std.max(),
                "mc_mse": mc_mse,
                "mc_estimates": mc_estimates.mean(),
                "mc_estimates_std": mc_estimates.std(),
                "mc_estimates_max": mc_estimates.max(),
                "mc_estimates_min": mc_estimates.min(),
            }

        # compute gradients and update params
        new_state, info, grad_norm = self.state.apply_loss_fns(
            loss_fn,
            pmap_axis=pmap_axis,
            has_aux=True,
            return_grad_norm=True,
        )
        info["grad_norm"] = grad_norm

        # log learning rates
        info["lr"] = self.lr_schedule(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames="argmax")
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax=False
    ) -> jnp.ndarray:
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (observations, goals),
            temperature=temperature,
            name="actor",
        )
        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            actions = dist.mode()
        else:
            actions = dist.sample(seed=seed)
        return actions

    @jax.jit
    def get_debug_metrics(self, batch, **kwargs):
        dist = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            temperature=1.0,
            name="actor",
        )
        pi_actions = dist.mode()
        log_probs = dist.log_prob(batch["actions"])
        mse = ((pi_actions - batch["actions"]) ** 2).sum(-1)

        mc_estimates = self.state.apply_fn(
            {"params": self.state.params},
            (batch["observations"], batch["goals"]),
            name="mc_return_regressor",
        )
        mc_mse = ((mc_estimates - batch["mc_returns"]) ** 2).mean()

        return {
            "actor_mse": mse,
            "actor_log_probs": log_probs,
            "actor_pi_actions": pi_actions,
            "mc_mse": mc_mse,
            "mc_estimates": mc_estimates.mean(),
            "mc_estimates_std": mc_estimates.std(),
            "mc_estimates_max": mc_estimates.max(),
            "mc_estimates_min": mc_estimates.min(),
        }

    def extract_encoder(self):
        return self.state.params["modules_actor"]["encoder"]["encoder"]

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        # example arrays for model init
        observations: FrozenDict,
        actions: jnp.ndarray,
        goals: FrozenDict,
        # agent config
        encoder_def: nn.Module,
        language_conditioned: bool = False,
        # should only be set if not language conditioned
        shared_goal_encoder: Optional[bool] = None,
        early_goal_concat: Optional[bool] = None,
        # other shared network config
        use_proprio: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": False,
            "std_parameterization": "exp",
        },
        # optimizer config
        learning_rate: float = 3e-4,
        warmup_steps: int = 1000,
        decay_steps: int = 1000000,
        # joint mc_return and gcbc loss
        gcbc_loss_ratio: float = 0.5,
        freeze_encoder: bool = False,
    ):
        if not language_conditioned:
            if shared_goal_encoder is None or early_goal_concat is None:
                raise ValueError(
                    "If not language conditioned, shared_goal_encoder and early_goal_concat must be set"
                )

            if early_goal_concat:
                # passing None as the goal encoder causes early goal concat
                goal_encoder_def = None
            else:
                if shared_goal_encoder:
                    goal_encoder_def = encoder_def
                else:
                    goal_encoder_def = copy.deepcopy(encoder_def)

            encoder_def = GCEncodingWrapper(
                encoder=encoder_def,
                goal_encoder=goal_encoder_def,
                use_proprio=use_proprio,
                stop_gradient=freeze_encoder,
            )
        else:
            if shared_goal_encoder is not None or early_goal_concat is not None:
                raise ValueError(
                    "If language conditioned, shared_goal_encoder and early_goal_concat must not be set"
                )
            encoder_def = LCEncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=freeze_encoder,
            )

        network_kwargs["activate_final"] = True
        networks = {
            "actor": Policy(
                encoder_def,
                MLP(**network_kwargs),
                action_dim=actions.shape[-1],
                **policy_kwargs
            ),
            "mc_return_regressor": ValueCritic(
                encoder_def,  # we're sharing the vision backbone
                MLP(**network_kwargs),
            ),
        }

        model_def = ModuleDict(networks)

        # create optimizer
        tx, lr_schedule = make_optimizer(
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            cosine_decay_steps=decay_steps if decay_steps is not None else None,
            weight_decay=0.001,
            beta2=0.98,
            clip_grad_norm=1.0,
            return_lr_schedule=True,
        )

        rng, init_rng = jax.random.split(rng)
        params = jax.jit(model_def.init)(
            init_rng,
            actor=[(observations, goals)],
            mc_return_regressor=[(observations, goals)],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                gcbc_loss_ratio=gcbc_loss_ratio,
            )
        )

        return cls(state, lr_schedule, config)
