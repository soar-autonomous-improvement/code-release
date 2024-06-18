"""
Implementation of CQL in continuous action spaces.
"""
import copy
from functools import partial
from typing import Optional, Tuple

import chex
import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from ml_collections import ConfigDict
from overrides import overrides

from jaxrl_m.agents.continuous.sac import SACAgent, _create_encoder_def
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import *
from jaxrl_m.networks.actor_critic_nets import (
    Critic,
    DistributionalCritic,
    Policy,
    ensemblize,
)
from jaxrl_m.networks.distributional import hl_gauss_transform
from jaxrl_m.networks.lagrange import GeqLagrangeMultiplier, LeqLagrangeMultiplier
from jaxrl_m.networks.mlp import MLP, LayerInputMLP


class ContinuousCQLAgent(SACAgent):
    @overrides
    def _sample_negative_goals(self, batch, rng):
        """for calql, adjust the mc_returns for negative goals"""
        new_stats, neg_goal_mask = super()._sample_negative_goals(batch, rng)
        if self.config["use_calql"]:
            assert "mc_returns" in batch
            new_mc_returns = jnp.where(
                neg_goal_mask, -1 / (1 - self.config["discount"]), batch["mc_returns"]
            )
            return {
                **new_stats,
                "mc_returns": new_mc_returns,
            }, neg_goal_mask
        else:
            return new_stats, neg_goal_mask

    def forward_cql_alpha_lagrange(self, *, grad_params: Optional[Params] = None):
        """
        Forward pass for the CQL alpha Lagrange multiplier
        Pass grad_params to use non-default parameters (e.g. for gradients).
        """
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            name="cql_alpha_lagrange",
        )

    def forward_policy_and_sample(
        self,
        obs: Data,
        rng: PRNGKey,
        *,
        grad_params: Optional[Params] = None,
        repeat=None,
    ):
        rng, sample_rng = jax.random.split(rng)
        action_dist = super().forward_policy(obs, rng, grad_params=grad_params)

        def sample_n_binarized_gripper(action_distributions, sample_rng, n=None):
            multiple_actions = n != None
            if not multiple_actions:
                n = 1

            # Sample actions, applying trick to sample only binary actions for gripper dimension
            actions = action_distributions.sample(seed=sample_rng, sample_shape=n)
            assert len(actions.shape) == 3
            last_dim_mean, last_dim_std = (
                action_distributions.mode()[:, -1],
                action_distributions.stddev()[:, -1],
            )
            # print("##########################")
            # print(last_dim_mean)
            # print()
            # print(last_dim_std)
            # print()
            actions_no_grad = jax.lax.stop_gradient(actions)
            last_dim_mean_no_grad, last_dim_std_no_grad = jax.lax.stop_gradient(
                last_dim_mean
            ), jax.lax.stop_gradient(last_dim_std)

            # Construct categorical distribution for two gripper actions: open and close
            open_action, close_action = 0.95, -0.95
            actions_gripper_open = jnp.concatenate(
                [
                    actions_no_grad[:, :, :-1],
                    open_action * jnp.ones((n, len(actions[0]), 1)),
                ],
                axis=-1,
            )
            # print(actions_gripper_open.shape)
            # print()
            log_probs_open = action_distributions.log_prob(actions_gripper_open)
            # print(log_probs_open)
            # print()
            actions_gripper_close = jnp.concatenate(
                [
                    actions_no_grad[:, :, :-1],
                    close_action * jnp.ones((n, len(actions[0]), 1)),
                ],
                axis=-1,
            )
            log_probs_close = action_distributions.log_prob(actions_gripper_close)
            # print(log_probs_close)
            # print()
            gripper_action_sampling_dist = distrax.Categorical(
                logits=jnp.concatenate(
                    [log_probs_open[..., None], log_probs_close[..., None]], axis=-1
                )
            )
            gripper_action_samples = gripper_action_sampling_dist.sample(
                seed=sample_rng
            )
            # print(gripper_action_samples)

            # Finish applying the trick
            gripper_action_samples = open_action * (
                gripper_action_samples == 0
            ) + close_action * (gripper_action_samples == 1)
            z = (gripper_action_samples - last_dim_mean_no_grad) / (
                last_dim_std_no_grad + 0.0000001
            )
            z = jax.lax.stop_gradient(z)  # For good measure
            # print(z)
            # print()
            last_dim_samples = last_dim_mean + last_dim_std * z
            actions = jnp.concatenate(
                [actions[:, :, :-1], last_dim_samples[..., None]], axis=-1
            )
            log_probs = action_distributions.log_prob(actions)

            if not multiple_actions:
                actions, log_probs = actions[0], log_probs[0]

            return actions, log_probs

        if repeat:
            new_actions, log_pi = action_dist.sample_and_log_prob(
                seed=sample_rng, sample_shape=repeat
            )
            # new_actions, log_pi = sample_n_binarized_gripper(action_dist, sample_rng, repeat)
            new_actions = jnp.transpose(
                new_actions, (1, 0, 2)
            )  # (batch, repeat, action_dim)
            log_pi = jnp.transpose(log_pi, (1, 0))  # (batch, repeat)
        else:
            new_actions, log_pi = action_dist.sample_and_log_prob(seed=sample_rng)
            # new_actions, log_pi = sample_n_binarized_gripper(action_dist, sample_rng, None)
        return new_actions, log_pi

    def _get_cql_q_diff(
        self, batch, rng: PRNGKey, grad_params: Optional[Params] = None
    ):
        """
        most of the CQL loss logic is here
        It is needed for both critic_loss_fn and cql_alpha_loss_fn
        """
        batch_size = batch["rewards"].shape[0]
        q_pred = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            batch["actions"],
            rng,
            grad_params=grad_params,
        )
        chex.assert_shape(q_pred, (self.config["critic_ensemble_size"], batch_size))

        """sample random actions"""
        action_dim = batch["actions"].shape[-1]
        rng, action_rng = jax.random.split(rng)
        if self.config["cql_action_sample_method"] == "uniform":
            cql_random_actions = jax.random.uniform(
                action_rng,
                shape=(batch_size, self.config["cql_n_actions"], action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        elif self.config["cql_action_sample_method"] == "normal":
            cql_random_actions = jax.random.normal(
                action_rng,
                shape=(batch_size, self.config["cql_n_actions"], action_dim),
            )
        else:
            raise NotImplementedError

        rng, current_a_rng, next_a_rng = jax.random.split(rng, 3)
        cql_current_actions, cql_current_log_pis = self.forward_policy_and_sample(
            self._include_goals_in_obs(batch, "observations"),
            current_a_rng,
            repeat=self.config["cql_n_actions"],
        )
        chex.assert_shape(
            cql_current_log_pis, (batch_size, self.config["cql_n_actions"])
        )

        cql_next_actions, cql_next_log_pis = self.forward_policy_and_sample(
            self._include_goals_in_obs(batch, "next_observations"),
            next_a_rng,
            repeat=self.config["cql_n_actions"],
        )

        all_sampled_actions = jnp.concatenate(
            [
                cql_random_actions,
                cql_current_actions,
                cql_next_actions,
            ],
            axis=1,
        )

        """q values of randomly sampled actions"""
        rng, q_rng = jax.random.split(rng)
        cql_q_samples = self.forward_critic(
            self._include_goals_in_obs(batch, "observations"),
            all_sampled_actions,  # this is being vmapped over in sac.py
            q_rng,
            grad_params=grad_params,
            train=True,
        )
        chex.assert_shape(
            cql_q_samples,
            (
                self.config["critic_ensemble_size"],
                batch_size,
                self.config["cql_n_actions"] * 3,
            ),
        )

        if self.config["critic_subsample_size"] is not None:
            rng, subsample_key = jax.random.split(rng)
            subsample_idcs = jax.random.randint(
                subsample_key,
                (self.config["critic_subsample_size"],),
                0,
                self.config["critic_ensemble_size"],
            )
            cql_q_samples = cql_q_samples[subsample_idcs]
            q_pred = q_pred[subsample_idcs]
            critic_size = self.config["critic_subsample_size"]
        else:
            critic_size = self.config["critic_ensemble_size"]
        """Cal-QL"""
        if self.config["use_calql"]:
            mc_lower_bound = jnp.repeat(
                batch["mc_returns"].reshape(-1, 1),
                self.config["cql_n_actions"] * 3,
                axis=1,
            )
            chex.assert_shape(
                mc_lower_bound, (batch_size, self.config["cql_n_actions"] * 3)
            )

            num_vals = jnp.size(cql_q_samples)
            calql_bound_rate = jnp.sum(cql_q_samples < mc_lower_bound) / num_vals
            cql_q_samples = jnp.maximum(cql_q_samples, mc_lower_bound)

        if self.config["cql_importance_sample"]:
            random_density = jnp.log(0.5**action_dim)

            importance_prob = jnp.concatenate(
                [
                    jnp.broadcast_to(
                        random_density, (batch_size, self.config["cql_n_actions"])
                    ),
                    cql_current_log_pis,
                    cql_next_log_pis,  # this order matters, should match all_sampled_actions
                ],
                axis=1,
            )
            cql_q_samples = cql_q_samples - importance_prob  # broadcast over dim 0
        else:
            cql_q_samples = jnp.concatenate(
                [
                    cql_q_samples,
                    jnp.expand_dims(q_pred, -1),
                ],
                axis=-1,
            )
            cql_q_samples -= jnp.log(cql_q_samples.shape[-1]) * self.config["cql_temp"]
            chex.assert_shape(
                cql_q_samples,
                (
                    critic_size,
                    batch_size,
                    3 * self.config["cql_n_actions"] + 1,
                ),
            )

        """log sum exp of the ood actions"""
        cql_ood_values = (
            jax.scipy.special.logsumexp(
                cql_q_samples / self.config["cql_temp"], axis=-1
            )
            * self.config["cql_temp"]
        )
        chex.assert_shape(cql_ood_values, (critic_size, batch_size))

        cql_q_diff = cql_ood_values - q_pred
        info = {
            "cql_ood_values": cql_ood_values.mean(),
        }
        if self.config["use_calql"]:
            info["calql_bound_rate"] = calql_bound_rate

        return cql_q_diff, info

    @overrides
    def _compute_next_actions(self, batch, rng):
        """
        compute the next actions but with repeat cql_n_actions times
        this should only be used when calculating critic loss using
        cql_max_target_backup
        """
        sample_n_actions = (
            self.config["cql_n_actions"]
            if self.config["cql_max_target_backup"]
            else None
        )
        next_actions, next_actions_log_probs = self.forward_policy_and_sample(
            self._include_goals_in_obs(batch, "next_observations"),
            rng,
            repeat=sample_n_actions,
        )
        return next_actions, next_actions_log_probs

    @overrides
    def _process_target_next_qs(self, target_next_qs, next_actions_log_probs):
        """add cql_max_target_backup option"""

        if self.config["cql_max_target_backup"]:
            max_target_indices = jnp.expand_dims(
                jnp.argmax(target_next_qs, axis=-1), axis=-1
            )
            target_next_qs = jnp.take_along_axis(
                target_next_qs, max_target_indices, axis=-1
            ).squeeze(-1)
            next_actions_log_probs = jnp.take_along_axis(
                next_actions_log_probs, max_target_indices, axis=-1
            ).squeeze(-1)

        target_next_qs = super()._process_target_next_qs(
            target_next_qs,
            next_actions_log_probs,
        )

        return target_next_qs

    @overrides
    def critic_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """add CQL loss on top of SAC loss"""
        if self.config["use_td_loss"]:
            td_loss, td_loss_info = super().critic_loss_fn(batch, params, rng)
        else:
            td_loss, td_loss_info = 0.0, {}

        cql_q_diff, cql_intermediate_results = self._get_cql_q_diff(batch, rng, params)

        """auto tune cql alpha"""
        if self.config["cql_autotune_alpha"]:
            alpha = self.forward_cql_alpha_lagrange()
            cql_loss = (cql_q_diff - self.config["cql_target_action_gap"]).mean()
        else:
            alpha = self.config["cql_alpha"]
            cql_loss = jnp.clip(
                cql_q_diff,
                self.config["cql_clip_diff_min"],
                self.config["cql_clip_diff_max"],
            ).mean()

        critic_loss = td_loss + alpha * cql_loss

        info = {
            **td_loss_info,
            "critic_loss": critic_loss,
            "td_loss": td_loss,
            "cql_loss": cql_loss,
            "cql_alpha": alpha,
            "cql_diff": cql_q_diff.mean(),
            **cql_intermediate_results,
        }

        return critic_loss, info

    def cql_alpha_lagrange_penalty(
        self, qvals_diff, *, grad_params: Optional[Params] = None
    ):
        return self.state.apply_fn(
            {"params": grad_params or self.state.params},
            lhs=qvals_diff,
            rhs=self.config["cql_target_action_gap"],
            name="cql_alpha_lagrange",
        )

    def cql_alpha_loss_fn(self, batch, params: Params, rng: PRNGKey):
        """recompute cql_q_diff without gradients (not optimal for runtime)"""
        cql_q_diff, _ = self._get_cql_q_diff(batch, rng)

        cql_alpha_loss = self.cql_alpha_lagrange_penalty(
            qvals_diff=cql_q_diff.mean(),
            grad_params=params,
        )
        lmbda = self.forward_cql_alpha_lagrange()

        return cql_alpha_loss, {
            "cql_alpha_loss": cql_alpha_loss,
            "cql_alpha_lagrange_multiplier": lmbda,
        }

    @overrides
    def loss_fns(self, batch):
        losses = super().loss_fns(batch)
        if self.config["cql_autotune_alpha"]:
            losses["cql_alpha_lagrange"] = partial(self.cql_alpha_loss_fn, batch)

        return losses

    def update(
        self,
        batch: Batch,
        pmap_axis: str = None,
        networks_to_update: set = set({"actor", "critic"}),
    ):
        """update super() to perhaps include updating CQL lagrange multiplier"""
        if self.config["cql_autotune_alpha"]:
            networks_to_update.add("cql_alpha_lagrange")

        return super().update(
            batch, pmap_axis=pmap_axis, networks_to_update=frozenset(networks_to_update)
        )

    def update_cql_alpha(self, new_alpha):
        """update the CQL alpha. Used for finetuning online with a different alpha"""
        object.__setattr__(
            self, "config", self.config.copy({"cql_alpha": new_alpha})
        )  # hacky way to update self.config because self is a frozen dataclass

    def update_bc_loss_weight(self, new_bc_loss_weight):
        """update the BC loss weight"""
        object.__setattr__(
            self, "config", self.config.copy({"bc_loss_weight": new_bc_loss_weight})
        )  # hacky way to update self.config because self is a frozen dataclass

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: Data,
        actions: jnp.ndarray,
        # Model arch
        encoder_def: nn.Module,
        shared_encoder: bool = False,
        stop_actor_encoder_gradient: bool = False,
        stop_critic_encoder_gradient: bool = False,
        distributional_critic: bool = False,
        distributional_critic_kwargs: dict = {
            "q_min": -100.0,
            "q_max": 0.0,
            "num_bins": 128,
        },
        critic_network_type: str = "mlp",
        critic_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        policy_network_kwargs: dict = {
            "hidden_dims": [256, 256],
            "activate_final": True,
            "use_layer_norm": False,
        },
        policy_kwargs: dict = {
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
        },
        # goals
        goals: Optional[Data] = None,
        early_goal_concat: bool = False,
        shared_goal_encoder: bool = True,
        **kwargs,
    ):
        # update algorithm config
        config = get_default_config(updates=kwargs)

        create_encoder_fn = partial(
            _create_encoder_def,
            use_proprio=False,
            enable_stacking=False,
            goal_conditioned=config.goal_conditioned,
            early_goal_concat=early_goal_concat,
            shared_goal_encoder=shared_goal_encoder,
        )

        if shared_encoder:
            encoder_def = create_encoder_fn(
                encoder_def=encoder_def, stop_gradient=False
            )
            encoders = {
                "actor": encoder_def,
                "critic": encoder_def,
            }
        else:
            encoders = {
                "actor": create_encoder_fn(
                    encoder_def=encoder_def,
                    stop_gradient=stop_actor_encoder_gradient,
                ),
                "critic": create_encoder_fn(
                    encoder_def=copy.deepcopy(encoder_def),
                    stop_gradient=stop_critic_encoder_gradient,
                ),
            }

        # Define networks
        policy_def = Policy(
            encoder=encoders["actor"],
            network=MLP(**policy_network_kwargs),
            action_dim=actions.shape[-1],
            **policy_kwargs,
            name="actor",
        )
        if critic_network_type == "mlp":
            critic_net = MLP
        elif critic_network_type == "ptr_critic":
            critic_net = LayerInputMLP
        else:
            raise NotImplementedError
        critic_backbone = partial(critic_net, **critic_network_kwargs)
        critic_backbone = ensemblize(critic_backbone, config.critic_ensemble_size)(
            name="critic_ensemble"
        )

        if distributional_critic:
            q_min = distributional_critic_kwargs["q_min"]
            q_max = distributional_critic_kwargs["q_max"]
            config["q_min"] = q_min
            config["q_max"] = q_max

            critic_def = partial(
                DistributionalCritic,
                encoder=encoders["critic"],
                network=critic_backbone,
                network_separate_action_input=critic_network_type == "ptr_critic",
                q_low=q_min,
                q_high=q_max,
                num_bins=distributional_critic_kwargs["num_bins"],
            )(name="critic")
            scalar_target_to_dist_fn = hl_gauss_transform(
                min_value=q_min,
                max_value=q_max,
                num_bins=distributional_critic_kwargs["num_bins"],
            )[0]
        else:
            critic_def = partial(
                Critic,
                encoder=encoders["critic"],
                network=critic_backbone,
                network_separate_action_input=critic_network_type == "ptr_critic",
            )(name="critic")
            scalar_target_to_dist_fn = None

        config["distributional_critic"] = distributional_critic
        config["scalar_target_to_dist_fn"] = scalar_target_to_dist_fn

        temperature_def = GeqLagrangeMultiplier(
            init_value=config.temperature_init,
            constraint_shape=(),
            name="temperature",
        )
        if config["cql_autotune_alpha"]:
            cql_alpha_lagrange_def = LeqLagrangeMultiplier(
                init_value=config.cql_alpha_lagrange_init,
                constraint_shape=(),
                name="cql_alpha_lagrange",
            )

        # model def
        networks = {
            "actor": policy_def,
            "critic": critic_def,
            "temperature": temperature_def,
        }
        if config["cql_autotune_alpha"]:
            networks["cql_alpha_lagrange"] = cql_alpha_lagrange_def
        if config["dr3_coefficient"] > 0:
            networks["critic_encoder"] = encoders["critic"]
        model_def = ModuleDict(networks)

        # Define optimizers
        txs = {
            "actor": make_optimizer(**config.actor_optimizer_kwargs),
            "critic": make_optimizer(**config.critic_optimizer_kwargs),
            "temperature": make_optimizer(**config.temperature_optimizer_kwargs),
        }
        if config["cql_autotune_alpha"]:
            txs["cql_alpha_lagrange"] = make_optimizer(
                **config.cql_alpha_lagrange_otpimizer_kwargs
            )

        # init params
        rng, init_rng = jax.random.split(rng)
        network_input = (
            (observations, goals) if config.goal_conditioned else observations
        )
        extra_kwargs = {}
        if config["cql_autotune_alpha"]:
            extra_kwargs["cql_alpha_lagrange"] = []
        if config["dr3_coefficient"] > 0:
            extra_kwargs["critic_encoder"] = [network_input]
        params = model_def.init(
            init_rng,
            actor=[network_input],
            critic=[network_input, actions],
            temperature=[],
            **extra_kwargs,
        )["params"]

        # create
        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=txs,
            target_params=params,
            rng=create_rng,
        )

        # config
        if config.target_entropy >= 0.0:
            config.target_entropy = -actions.shape[-1]
        config = flax.core.FrozenDict(config)

        return cls(state, config)


def get_default_config(updates=None):
    config = ConfigDict()
    config.discount = 0.99
    config.backup_entropy = False
    config.target_entropy = 0.0
    config.soft_target_update_rate = 5e-3
    config.critic_ensemble_size = 2
    config.critic_subsample_size = None
    config.autotune_entropy = True
    config.temperature_init = 1.0
    config.actor_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
            "warmup_steps": 0,
        }
    )
    config.critic_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
            "warmup_steps": 0,
        }
    )
    config.temperature_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )

    config.cql_n_actions = 4
    config.cql_action_sample_method = "uniform"
    config.cql_max_target_backup = True
    config.cql_importance_sample = True
    config.cql_autotune_alpha = False
    config.cql_alpha_lagrange_init = 1.0
    config.cql_alpha_lagrange_otpimizer_kwargs = ConfigDict(
        {
            "learning_rate": 3e-4,
        }
    )
    config.cql_target_action_gap = 1.0
    config.cql_temp = 1.0
    config.cql_alpha = 5.0
    config.cql_clip_diff_min = -np.inf
    config.cql_clip_diff_max = np.inf
    config.use_td_loss = True  # set this to False to essentially do BC

    # Cal-QL
    config.use_calql = False

    # AWAC
    config.use_awac = False

    # DR3 explicit regularization
    config.dr3_coefficient = 0.0

    # Goal-conditioning
    config.goal_conditioned = False
    config.gc_kwargs = ConfigDict(
        {
            "negative_proportion": 0.0,
        }
    )

    config.early_goal_concat = False

    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    return config
