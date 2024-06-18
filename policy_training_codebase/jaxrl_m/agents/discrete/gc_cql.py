"""Implementations of goal-conditioned CQL in discrete action spaces."""
import copy
import functools
from functools import partial

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from jaxrl_m.agents.discrete.cql import cql_loss_fn
from jaxrl_m.common.common import JaxRLTrainState, ModuleDict, nonpytree_field
from jaxrl_m.common.typing import *
from jaxrl_m.networks.discrete_nets import DiscreteCriticHead


class DiscreteGCQ(nn.Module):
    encoder: nn.Module
    goal_encoder: nn.Module
    qfunc: nn.Module

    def __call__(self, observations, goals):
        latents = self.encoder(observations)
        goal_latents = self.goal_encoder(goals)
        return self.qfunc(latents, goal_latents)


class GCAdaptor(nn.Module):
    """just a wrapper to make any network compatible with goal-conditioned policy"""

    network: nn.Module

    def __call__(self, observations, goals):
        combined = jnp.concatenate([observations, goals], axis=-1)
        return self.network(combined)


class GoalConditionedCQLAgent(flax.struct.PyTreeNode):
    model: JaxRLTrainState
    config: dict = nonpytree_field()
    lr_schedules: dict = nonpytree_field()

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
            q_target = batch["rewards"] + self.config["discount"] * nv * batch["masks"]

            # Current Q
            rng, dropout_rng = jax.random.split(rng)
            q = self.state.apply_fn(
                {"params": params},  # gradients flow through here
                batch["observations"],
                batch["goals"],
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

        # TODO: log learning rate if we use a lr schedule

        return self.replace(state=new_state), info

    @partial(jax.jit, static_argnames=("argmax"))
    def sample_actions(
        self,
        observations: np.ndarray,
        goals: np.ndarray,
        *,
        seed: PRNGKey,
        temperature: float = 1.0,
        argmax: bool = False,
    ) -> jnp.ndarray:
        logits = self.state.apply_fn(
            {"params": self.state.params},
            observations,
            goals,
            name="q_network",
        )
        dist = distrax.Categorical(logits=logits / temperature)

        if argmax:
            return dist.mode()
        else:
            return dist.sample(seed=seed)

    @classmethod
    def create(
        cls,
        rng: PRNGKey,
        observations: jnp.ndarray,
        goals: jnp.ndarray,
        n_actions: int,
        # Model architecture
        encoder_def: nn.Module,
        shared_goal_encoder: bool = False,
        network_kwargs: dict = {
            "hidden_dims": [256, 256],
        },
        optim_kwargs: dict = {
            "learning_rate": 6e-5,
        },
        # Algorithm config
        discount=0.95,
        cql_alpha=1.0,
        temperature=1.0,
        target_update_rate=0.002,
        **kwargs,
    ):
        if shared_goal_encoder:
            goal_encoder_def = encoder_def
        else:
            goal_encoder_def = copy.deepcopy(encoder_def)

        # for now, we only support Q-learning styled GC-CQL
        q_func = GCAdaptor(DiscreteCriticHead(n_actions=n_actions, **network_kwargs))
        networks = {
            "q_network": DiscreteGCQ(
                encoder=encoder_def, goal_encoder=goal_encoder_def, qfunc=q_func
            ),
        }

        model_def = ModuleDict(networks)

        # TODO: add learning rate decay schedule
        lr_schedule = None
        tx = optax.adam(**optim_kwargs)

        rng, init_rng = jax.random.split(rng)
        params = model_def.init(
            init_rng,
            q_network=[(observations, goals)],
        )["params"]

        rng, create_rng = jax.random.split(rng)
        state = JaxRLTrainState.create(
            apply_fn=model_def.apply,
            params=params,
            txs=tx,
            target_params=params,
            rng=create_rng,
        )

        config = flax.core.FrozenDict(
            dict(
                discount=discount,
                cql_alpha=cql_alpha,
                temperature=temperature,
                target_update_rate=target_update_rate,
            )
        )

        return cls(state, config, lr_schedule)


def get_default_config():
    config = ml_collections.ConfigDict(
        {
            "algo": "gccql",
            "shared_goal_encoder": False,
            "optim_kwargs": {"learning_rate": 6e-5, "eps": 0.00015},
            "network_kwargs": {
                "hidden_dims": (256, 256),
            },
            "discount": 0.95,
            "cql_alpha": 0.5,
            "temperature": 1.0,
            "target_update_rate": 0.002,
        }
    )
    return config
