import copy
from functools import partial
from typing import Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core import FrozenDict

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.encoding import GCEncodingWrapper, LCEncodingWrapper
from jaxrl_m.common.typing import Batch, PRNGKey
from jaxrl_m.networks.diffusion_nets import (
    FourierFeatures,
    ScoreActor,
    cosine_beta_schedule,
    vp_beta_schedule,
)
from jaxrl_m.networks.mlp import MLP, MLPResNet


def ddpm_bc_loss(noise_prediction, noise):
    ddpm_loss = jnp.square(noise_prediction - noise).sum(-1)

    return ddpm_loss.mean(), {
        "ddpm_loss": ddpm_loss,
        "ddpm_loss_mean": ddpm_loss.mean(),
    }


class GCDDPMBCAgent(flax.struct.PyTreeNode):
    """
    Models action distribution with a diffusion model.

    Assumes observation histories as input and action sequences as output.
    """

    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

    def _sample_negative_goals(self, batch):
        """
        for goal/reward relabeling
        sample negative goals and change the batch rewards
        """
        batch_size = batch["rewards"].shape[0]
        neg_goal_indices = jnp.roll(jnp.arange(batch_size, dtype=jnp.int32), -1)

        neg_goal_mask = (jnp.arange(batch_size) / batch_size) < self.config[
            "gc_kwargs"
        ]["negative_proportion"]
        goal_indices = jnp.where(
            neg_goal_mask, neg_goal_indices, jnp.arange(batch_size)
        )
        new_goals = jax.tree_map(lambda x: x[goal_indices], batch["goals"])
        new_rewards = jnp.where(neg_goal_mask, -1, batch["rewards"])
        new_mc_returns = jnp.where(neg_goal_mask, -50.0, batch["mc_returns"])
        new_masks = jnp.where(neg_goal_mask, jnp.ones_like(batch["masks"]), batch["masks"]) 

        # save the goals and rewards
        batch["rewards"] = new_rewards
        batch["goals"] = new_goals
        batch["mc_returns"] = new_mc_returns
        batch["masks"] = new_masks

    @partial(jax.jit, static_argnames="pmap_axis")
    def update(self, batch: Batch, pmap_axis: str = None):
        def actor_loss_fn(params, rng):
            key, rng = jax.random.split(rng)
            time = jax.random.randint(
                key, (batch["actions"].shape[0],), 0, self.config["diffusion_steps"]
            )
            key, rng = jax.random.split(rng)
            noise_sample = jax.random.normal(key, batch["actions"].shape)

            alpha_hats = self.config["alpha_hats"][time]
            time = time[:, None]
            alpha_1 = jnp.sqrt(alpha_hats)[:, None, None]
            alpha_2 = jnp.sqrt(1 - alpha_hats)[:, None, None]

            noisy_actions = alpha_1 * batch["actions"] + alpha_2 * noise_sample

            rng, key = jax.random.split(rng)
            noise_pred = self.state.apply_fn(
                {"params": params},  # gradient flows through here
                (batch["observations"], batch["goals"]),
                noisy_actions,
                time,
                #batch["mc_returns"] / 25 + 1,
                train=True,
                rngs={"dropout": key},
                name="actor",
            )

            return ddpm_bc_loss(
                noise_pred,
                noise_sample,
            )

        # Sample negative goals
        #self._sample_negative_goals(batch)

        loss_fns = {
            "actor": actor_loss_fn,
        }

        # compute gradients and update params
        new_state, info, grad_norm = self.state.apply_loss_fns(
            loss_fns, pmap_axis=pmap_axis, has_aux=True, return_grad_norm=True,
        )
        info["grad_norm"] = grad_norm

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # log learning rates
        info["actor_lr"] = self.lr_schedules["actor"](self.state.step)

        return self.replace(state=new_state), info

    @jax.jit
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        #mc_returns: np.ndarray,
        *,
        seed: PRNGKey = None,
        temperature: float = 1.0,
        clip_sampler: bool = True,
    ) -> jnp.ndarray:
        assert len(observations["image"].shape) > 3, "Must use observation histories"

        def fn(input_tuple, time):
            current_x, rng = input_tuple
            input_time = jnp.broadcast_to(time, (current_x.shape[0], 1))

            eps_pred = self.state.apply_fn(
                {"params": self.state.target_params},
                (observations, goals),
                current_x,
                input_time,
        #        mc_returns / 25 + 1,
                name="actor",
            )

            alpha_1 = 1 / jnp.sqrt(self.config["alphas"][time])
            alpha_2 = (1 - self.config["alphas"][time]) / (
                jnp.sqrt(1 - self.config["alpha_hats"][time])
            )
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            rng, key = jax.random.split(rng)
            z = jax.random.normal(
                key,
                shape=current_x.shape,
            )
            z_scaled = temperature * z
            current_x = current_x + (time > 0) * (
                jnp.sqrt(self.config["betas"][time]) * z_scaled
            )

            if clip_sampler:
                current_x = jnp.clip(
                    current_x, self.config["action_min"], self.config["action_max"]
                )

            return (current_x, rng), ()

        key, rng = jax.random.split(seed)

        if len(observations["image"].shape) == 4:
            # unbatched input from evaluation
            batch_size = 1
            observations = jax.tree_map(lambda x: x[None], observations)
            goals = jax.tree_map(lambda x: x[None], goals)
        else:
            batch_size = observations["image"].shape[0]

        input_tuple, () = jax.lax.scan(
            fn,
            (jax.random.normal(key, (batch_size, *self.config["action_dim"])), rng),
            jnp.arange(self.config["diffusion_steps"] - 1, -1, -1),
        )

        for _ in range(self.config["repeat_last_step"]):
            input_tuple, () = fn(input_tuple, 0)

        action_0, rng = input_tuple

        if batch_size == 1:
            # this is an evaluation call so unbatch
            return action_0[0]
        else:
            return action_0

    @jax.jit
    def get_debug_metrics(self, batch, seed, gripper_close_val=None):
        actions = self.sample_actions(
            observations=batch["observations"], goals=batch["goals"], seed=seed
            #observations=batch["observations"], goals=batch["goals"], mc_returns=batch["mc_returns"], seed=seed
        )

        metrics = {
            "mse": ((actions - batch["actions"]) ** 2).sum((-2, -1)).mean(),
        }

        return metrics

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        # example arrays for model init
        observations: FrozenDict,
        goals: FrozenDict,
        actions: jnp.ndarray,
        # agent config
        encoder_def: nn.Module,
        language_conditioned: bool = False,
        # should only be set if not language conditioned
        shared_goal_encoder: Optional[bool] = None,
        early_goal_concat: Optional[bool] = None,
        # other shared network config
        use_proprio: bool = False,
        score_network_kwargs: dict = {
            "time_dim": 32,
            "num_blocks": 3,
            "dropout_rate": 0.1,
            "hidden_dim": 256,
        },
        gc_kwargs: dict = {
            "negative_proportion": 0.0,
        },
        # optimizer
        learning_rate: float = 3e-4,
        warmup_steps: int = 2000,
        actor_decay_steps: Optional[int] = None,
        # DDPM algorithm train + inference config
        beta_schedule: str = "cosine",
        diffusion_steps: int = 25,
        action_samples: int = 1,
        repeat_last_step: int = 0,
        target_update_rate=0.002,
        dropout_target_networks=True,
    ):
        assert len(actions.shape) > 1, "Must use action chunking"
        assert len(observations["image"].shape) > 3, "Must use observation histories"

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
                stop_gradient=False,
            )
        else:
            if shared_goal_encoder is not None or early_goal_concat is not None:
                raise ValueError(
                    "If language conditioned, shared_goal_encoder and early_goal_concat must not be set"
                )
            encoder_def = LCEncodingWrapper(
                encoder=encoder_def,
                use_proprio=use_proprio,
                stop_gradient=False,
            )

        networks = {
            "actor": ScoreActor(
                encoder_def,
                FourierFeatures(score_network_kwargs["time_dim"], learnable=True),
                MLP(
                    (
                        2 * score_network_kwargs["time_dim"],
                        score_network_kwargs["time_dim"],
                    )
                ),
                MLPResNet(
                    score_network_kwargs["num_blocks"],
                    actions.shape[-2] * actions.shape[-1],
                    dropout_rate=score_network_kwargs["dropout_rate"],
                    use_layer_norm=score_network_kwargs["use_layer_norm"],
                ),
            ),
        }

        model_def = ModuleDict(networks)

        rng, init_rng = jax.random.split(rng)
        if len(actions.shape) == 3:
            example_time = jnp.zeros((actions.shape[0], 1))
        else:
            example_time = jnp.zeros((1,))
        params = jax.jit(model_def.init)(
            init_rng, actor=[(observations, goals), actions, example_time]
            #init_rng, actor=[(observations, goals), actions, example_time, jnp.zeros((len(actions)))]
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
        txs = {
            "actor": actor_optimizer,
        }
        lr_schedules = {
            "actor": actor_lr_schedule,
        }

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        if beta_schedule == "cosine":
            betas = jnp.array(cosine_beta_schedule(diffusion_steps))
        elif beta_schedule == "linear":
            betas = jnp.linspace(1e-4, 2e-2, diffusion_steps)
        elif beta_schedule == "vp":
            betas = jnp.array(vp_beta_schedule(diffusion_steps))

        alphas = 1 - betas
        alpha_hat = jnp.array(
            [jnp.prod(alphas[: i + 1]) for i in range(diffusion_steps)]
        )

        config = flax.core.FrozenDict(
            dict(
                target_update_rate=target_update_rate,
                dropout_target_networks=dropout_target_networks,
                action_dim=actions.shape[-2:],
                action_max=2.0,
                action_min=-2.0,
                betas=betas,
                alphas=alphas,
                alpha_hats=alpha_hat,
                diffusion_steps=diffusion_steps,
                action_samples=action_samples,
                repeat_last_step=repeat_last_step,
                gc_kwargs=gc_kwargs,
            )
        )
        return cls(state, config, lr_schedules)
