"""Implementations of CQL in discrete action spaces."""
from functools import partial
from typing import Optional

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.encoding import EncodingWrapper
from jaxrl_m.common.optimizers import make_optimizer
from jaxrl_m.common.typing import *
from jaxrl_m.envs.discretize_continuous_action import (
    q_transformer_action_discretization,
    q_transformer_action_discretization_init,
    q_transformer_choose_actions,
)
from jaxrl_m.networks.discrete_nets import DiscreteCriticHead, DiscreteQ
from jaxrl_m.utils.timer_utils import Timer


def cql_loss_fn(
    q, q_pred, q_target, cql_temperature=1.0, cql_alpha=1.0, use_only_cql_loss=False
):

    cql_loss = (
        jax.scipy.special.logsumexp(q / cql_temperature, axis=-1)
        - q_pred / cql_temperature
    )
    if use_only_cql_loss:
        td_loss = jnp.zeros_like(q_pred)
        critic_loss = cql_loss
    else:
        td_loss = jnp.square(q_pred - q_target)
        critic_loss = td_loss + cql_alpha * cql_loss

    dist = distrax.Categorical(logits=q / cql_temperature)
    q_sorted = jnp.sort(q, axis=-1)

    return critic_loss.mean(), {
        "critic_loss": critic_loss.mean(),
        "td_loss": td_loss.mean(),
        "cql_loss": cql_loss.mean(),
        "td_loss max": td_loss.max(),
        "td_loss min": td_loss.min(),
        "qfunc entropy": dist.entropy().mean(),
        "q": q_pred.mean(),
        "q_pi": jnp.max(q, axis=-1).mean(),
        "target_q": q_target.mean(),
        "q_gap": jnp.mean(q_sorted[:, -1] - q_sorted[:, -2]),
        "q_gap max": jnp.max(q_sorted[:, -1] - q_sorted[:, -2]),
        "q_gap min": jnp.min(q_sorted[:, -1] - q_sorted[:, -2]),
    }


class DiscreteCQLAgent(flax.struct.PyTreeNode):
    state: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()
    timer = Timer()

    @partial(jax.jit, static_argnames=("pmap_axis",))
    def update(self, batch: Batch, pmap_axis: str = None):
        def loss_fn(params, rng):
            # Target Q
            rng, dropout_rng = jax.random.split(rng)
            target_nq = self.state.apply_fn(
                {"params": self.state.target_params},  # no gradient flows through here
                batch["next_observations"],
                rngs={"dropout": dropout_rng},
                name="q_network",
            )
            nq = self.state.apply_fn(
                {"params": self.state.params},  # no graident flows through here
                batch["next_observations"],
                rngs={"dropout": dropout_rng},
                name="q_network",
            )
            greedy_actions = jnp.argmax(nq, axis=-1)
            nv = target_nq[jnp.arange(len(batch["actions"])), greedy_actions]

            discounts = (
                batch["discounts"] if "discounts" in batch else self.config["discount"]
            )  # used so that Q-transformer can change some discounts to 1
            q_target = batch["rewards"] + discounts * nv * batch["masks"]

            # Current Q
            rng, dropout_rng = jax.random.split(rng)
            q = self.state.apply_fn(
                {"params": params},  # gradients flow through here
                batch["observations"],
                rngs={"dropout": dropout_rng},
                name="q_network",
            )
            q_pred = q[jnp.arange(len(batch["actions"])), batch["actions"]]

            # CQL Loss
            critic_loss, info = cql_loss_fn(
                q,
                q_pred,
                q_target,
                cql_temperature=self.config["temperature"],
                cql_alpha=self.config["cql_alpha"],
                use_only_cql_loss=self.config["use_only_cql_loss"],
            )
            return critic_loss, info

        # compute gradients and update params
        new_state, info = self.state.apply_loss_fns(
            loss_fn,
            pmap_axis=pmap_axis,
            has_aux=True,
        )

        # update the target params
        new_state = new_state.target_update(self.config["target_update_rate"])

        # update learning rate
        info["lr"] = self.lr_schedules(self.state.step)

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax: bool = False,
    ) -> jnp.ndarray:
        logits = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            name="q_network",
        )
        dist = distrax.Categorical(logits=logits / temperature)

        if argmax:
            assert seed is None, "Cannot specify seed when sampling deterministically"
            return dist.mode()
        else:
            if self.config["cql_alpha"] == 0:
                # epsilon greedy exploration for DoubleDQN
                seed, epsilon_rng, randint_rng = jax.random.split(seed, 3)
                actions = jnp.where(
                    jax.random.bernoulli(
                        epsilon_rng, p=self.config["dqn_exploration_epsilon"]
                    ),
                    jax.random.randint(
                        randint_rng,
                        minval=0,
                        maxval=self.config["n_actions"],
                        shape=(observations.shape[0],),
                    ),
                    dist.sample(seed=seed),
                )
            else:
                actions = dist.sample(seed=seed)
            return actions

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: jnp.ndarray,
        n_actions: int,
        # Model architecture
        encoder_def: nn.Module,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        **kwargs,
    ):
        config = get_default_config(updates=kwargs)
        config.n_actions = n_actions

        encoder_def = EncodingWrapper(
            encoder=encoder_def,
            use_proprio=False,
            stop_gradient=False,
        )

        # This is Q-learning styled CQL, Actor-critic styl CQL is in agents/continuous
        q_func = DiscreteCriticHead(n_actions=n_actions, **network_kwargs)
        networks = {
            "q_network": DiscreteQ(encoder=encoder_def, qfunc=q_func),
        }

        model_def = ModuleDict(networks)

        tx, lr_schedule = make_optimizer(
            learning_rate=config.learning_rate,
            warmup_steps=config.optim_warmup_steps,
            clip_grad_norm=config.optim_clip_grad_max,
            return_lr_schedule=True,
        )

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            q_network=observations,
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(config)

        return cls(state, config, lr_schedule)


class QTransformerDiscretizedCQLAgent(DiscreteCQLAgent):
    """
    A CQL agent whose action space is discretized using the Q-transformer method.
    Everything is the same as the DiscreteCQL agent, except that we call the
    q_transformer discretization methods on the update() and action_prediction() methods.
    """

    def update(self, batch: Batch, pmap_axis: str = None):
        """process the batch to discretize it before calling the real update function"""
        with self.timer.context("q_transformer_discretize"):
            per_dim_batch = q_transformer_action_discretization(batch)
        critic_loss, info = super().update(per_dim_batch, pmap_axis=pmap_axis)
        time_taken = self.timer.get_average_times()
        info = {**info, "time": time_taken}
        return critic_loss, info

    def sample_actions(
        self,
        observations: np.ndarray,
        *,
        seed: Optional[PRNGKey] = None,
        temperature: float = 1.0,
        argmax: bool = False,
    ) -> jnp.ndarray:
        return q_transformer_choose_actions(
            observations,
            agent=super(),
            rng=seed,
            temperature=temperature,
            argmax=argmax,
        )

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: jnp.ndarray,
        n_actions: int,
        # Model architecture
        encoder_def: nn.Module,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        **kwargs,
    ):
        """This function expects an argument q_transformer_kwargs, which is a dict"""

        assert "q_transformer_kwargs" in kwargs
        q_transformer_kwargs = kwargs["q_transformer_kwargs"]
        q_transformer_action_discretization_init(
            action_space=q_transformer_kwargs["action_space"],
            num_bins_per_action_dim=q_transformer_kwargs["num_bins_per_action_dim"],
            discount=q_transformer_kwargs["discount"],
        )

        return super().create(
            rng,
            observations,
            n_actions,
            encoder_def,
            network_kwargs,
            **kwargs,
        )


def get_default_config(updates=None):
    config = ml_collections.ConfigDict(
        {
            "use_only_cql_loss": False,
            "learning_rate": 6e-5,
            "optim_warmup_steps": 0,
            "optim_clip_grad_max": np.inf,
            "network_kwargs": {
                "hidden_dims": (256, 256),
            },
            "discount": 0.95,
            "cql_alpha": 0.5,
            "temperature": 1.0,
            "dqn_exploration_epsilon": 0.05,
            "target_update_rate": 0.002,
        }
    )

    if updates:
        config.update(ml_collections.ConfigDict(updates).copy_and_resolve_references())
    return config
