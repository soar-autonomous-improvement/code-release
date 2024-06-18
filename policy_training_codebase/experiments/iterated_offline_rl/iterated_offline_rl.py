import random
import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
import traceback
import wandb
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags
import os
import random
import copy

from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.bridge_dataset import BridgeDataset, glob_to_path_list
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders

try:
    from jax_smi import initialise_tracking  # type: ignore
    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "", "Experiment name.")
flags.DEFINE_string("project", None, "Wandb project name.")
flags.DEFINE_list('tag', list(), 'Name of experiment')
flags.DEFINE_string('group', None, 'Group of the wandb experiments')
flags.DEFINE_bool("debug", False, "Debug config")
flags.DEFINE_integer("max_episode_per_round", None, "Max number of success episodes per round")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "vision_backbone_config",
    None,
    "File path to the vision backbone hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "bridgedata_config",
    None,
    "File path to the bridgedata configuration.",
    lock_config=False,
)

def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)
    assert FLAGS.config.batch_size % num_devices == 0

    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": f"jaxrl_iterated_{FLAGS.config.agent}_bridge",
            "entity": "",
            "exp_descriptor": FLAGS.exp_name,
            "tag": FLAGS.tag,
            "group": FLAGS.group,
        }
    )
    if FLAGS.project is not None:
        wandb_config["project"] = FLAGS.project
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        debug=FLAGS.debug,
    )

    save_dir = tf.io.gfile.join(
        FLAGS.config.save_dir,
        wandb_logger.config.project,
        f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",
    )

    # load datasets
    random.seed(FLAGS.config.seed)
    assert FLAGS.bridgedata_config.sampling_weights.pretraining_data + \
        FLAGS.bridgedata_config.sampling_weights.autonomous_data_successes + \
        FLAGS.bridgedata_config.sampling_weights.autonomous_data_failures == 1.0
    bridge_data_paths = []

    for prefix in FLAGS.bridgedata_config.pretraining_data:
        bridge_data_paths += glob_to_path_list(["?*"], prefix=prefix, exclude=[])
    # Filter out everything that is not a tfrecord
    bridge_data_paths_filtered = []
    for path in bridge_data_paths:
        if "tfrecord" in path:
            bridge_data_paths_filtered.append(path)
    bridge_data_paths = bridge_data_paths_filtered
    random.shuffle(bridge_data_paths)

    # take 10 trajectories from bridge for the validation set
    val_paths = [bridge_data_paths[:10]]
    if FLAGS.bridgedata_config.sampling_weights.pretraining_data > 0:
        train_paths = [bridge_data_paths[10:]]
    else:
        train_paths = []

    if FLAGS.bridgedata_config.sampling_weights.autonomous_data_successes > 0:
        autonomous_success_paths = []
        for subfolder in FLAGS.bridgedata_config.autonomous_data_successes:
            # We will limit the total number of successes extracted from each subfolder
            subfolder_success_paths = []
            for prefix in subfolder:
                subfolder_success_paths += glob_to_path_list(["?*"], prefix=prefix, exclude=[])
            if FLAGS.max_episode_per_round is not None:
                autonomous_success_paths += subfolder_success_paths[:FLAGS.max_episode_per_round]
            else:
                autonomous_success_paths += subfolder_success_paths
        random.shuffle(autonomous_success_paths)
        train_paths.append(autonomous_success_paths)

    if FLAGS.bridgedata_config.sampling_weights.autonomous_data_failures > 0:
        autonomous_failure_paths = []
        for prefix in FLAGS.bridgedata_config.autonomous_data_failures:
            autonomous_failure_paths += glob_to_path_list(["?*"], prefix=prefix, exclude=[])
        random.shuffle(autonomous_failure_paths)
        train_paths.append(autonomous_failure_paths)

    # Create the data configuration for the training dataset
    train_dataset_kwargs = copy.deepcopy(FLAGS.config.dataset_kwargs)
    train_dataset_kwargs["goal_relabeling_strategy"] = "delta_goals"
    if FLAGS.bridgedata_config.sampling_weights.autonomous_data_successes == 0.0 and FLAGS.bridgedata_config.sampling_weights.autonomous_data_failures == 0.0:
        # We're not including autonomous data
        train_dataset_kwargs["goal_relabeling_kwargs"] = {
            "goal_delta": [
                0,
                FLAGS.bridgedata_config.uniform_goal_sampling_upper_limits["pretraining_data"]
            ]
        }
        sample_weights = None
    else:
        train_dataset_kwargs["goal_relabeling_kwargs"] = []
        sample_weights = []

        if FLAGS.bridgedata_config.sampling_weights.pretraining_data > 0.0:
            train_dataset_kwargs["goal_relabeling_kwargs"].append(
                {
                    "goal_delta": [
                        0,
                        FLAGS.bridgedata_config.uniform_goal_sampling_upper_limits["pretraining_data"]
                    ]
                },
            )
            sample_weights.append(
                FLAGS.bridgedata_config.sampling_weights["pretraining_data"]
            )

        if FLAGS.bridgedata_config.sampling_weights.autonomous_data_successes > 0.0:
            train_dataset_kwargs["goal_relabeling_kwargs"].append(
                {
                    "goal_delta": [
                        0,
                        FLAGS.bridgedata_config.uniform_goal_sampling_upper_limits["autonomous_data_successes"]
                    ]
                }
            )
            sample_weights.append(
                FLAGS.bridgedata_config.sampling_weights["autonomous_data_successes"]
            )

        if FLAGS.bridgedata_config.sampling_weights.autonomous_data_failures > 0.0:
            train_dataset_kwargs["goal_relabeling_kwargs"].append(
                {
                    "goal_delta": [
                        0,
                        FLAGS.bridgedata_config.uniform_goal_sampling_upper_limits["autonomous_data_failures"]
                    ]
                }
            )
            sample_weights.append(
                FLAGS.bridgedata_config.sampling_weights["autonomous_data_failures"]
            )

    # Create the data configuration for the validation dataset
    val_dataset_kwargs = copy.deepcopy(FLAGS.config.dataset_kwargs)
    val_dataset_kwargs["goal_relabeling_strategy"] = "delta_goals"
    val_dataset_kwargs["goal_relabeling_kwargs"] = {
        "goal_delta": [
            0,
            FLAGS.bridgedata_config.uniform_goal_sampling_upper_limits["pretraining_data"]
        ]
    }

    # If we are doing RL, add in the discount factor to the goal relabeling kwargs for both the training and validation configurations
    if FLAGS.config.agent in ["stable_contrastive_rl", "gc_iql"]:
        if type(train_dataset_kwargs["goal_relabeling_kwargs"]) == list:
            for i in range(len(train_dataset_kwargs["goal_relabeling_kwargs"])):
                train_dataset_kwargs["goal_relabeling_kwargs"][i]["discount"] = FLAGS.config.agent_kwargs.discount
        else:
            train_dataset_kwargs["goal_relabeling_kwargs"]["discount"] = FLAGS.config.agent_kwargs.discount
        val_dataset_kwargs["goal_relabeling_kwargs"]["discount"] = FLAGS.config.agent_kwargs.discount

    # Create the training dataset
    data_with_old_tf_format = []
    i_dataset = 0
    if FLAGS.bridgedata_config.sampling_weights.pretraining_data > 0.0:
        i_dataset += 1
    if FLAGS.bridgedata_config.sampling_weights.autonomous_data_successes > 0.0:
        data_with_old_tf_format.append(i_dataset)
        i_dataset += 1
    if FLAGS.bridgedata_config.sampling_weights.autonomous_data_failures > 0.0:
        data_with_old_tf_format.append(i_dataset)
        i_dataset += 1
    train_data = BridgeDataset(
        train_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        train=True,
        action_proprio_metadata=FLAGS.bridgedata_config.action_proprio_metadata,
        sample_weights=sample_weights,
        paths_index_with_old_tf_format=data_with_old_tf_format,
        **train_dataset_kwargs,
    )
    train_data_iter = map(shard_fn, train_data.iterator())

    # Create the validation set
    val_data = BridgeDataset(
        val_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        train=True, # this creates a shuffle buffer. w/o it we are evaluating on trajs from the same scene
        action_proprio_metadata=FLAGS.bridgedata_config.action_proprio_metadata,
        sample_weights=None,
        **val_dataset_kwargs,
    )

    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )

    # define encoder
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )
    if FLAGS.config.get("resume_path", "") != "":
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)

    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps - agent.state.step))):
        try:
            timer.tick("total")

            timer.tick("dataset")
            batch = shard_batch(next(train_data_iter), sharding)
            timer.tock("dataset")

            timer.tick("train")
            agent, update_info = agent.update(batch)
            timer.tock("train")

            if agent.state.step % FLAGS.config.eval_interval == 0:
                # validation data debug metrics
                logging.info("Validation...")
                timer.tick("val")
                val_metrics = []
                j = 0
                val_iter = map(shard_fn, val_data.iterator())
                for val_batch in val_iter:
                    rng, val_rng = jax.random.split(rng)
                    val_metrics.append(agent.get_debug_metrics(val_batch, seed=val_rng))
                    j += 1
                    if j >= FLAGS.config.num_val_batches:
                        break
                val_metrics = jax.tree_map(lambda *xs: np.mean(xs), *val_metrics)
                wandb_logger.log({"validation": val_metrics}, step=agent.state.step)
                timer.tock("val")

            if agent.state.step % FLAGS.config.save_interval == 0:
                logging.info("Saving checkpoint...")
                checkpoint_path = checkpoints.save_checkpoint(
                    save_dir, agent, step=agent.state.step, keep=1e7
                )
                logging.info("Saved checkpoint to %s", checkpoint_path)

            timer.tock("total")

            if agent.state.step % FLAGS.config.log_interval == 0:
                update_info = jax.device_get(update_info)
                wandb_logger.log({"training": update_info}, step=agent.state.step)

                wandb_logger.log({"timer": timer.get_average_times()}, step=agent.state.step)
        except tf.errors.OpError as e:
            # sometimes tfds will have trouble communicating with cloud storage bucket for some reason...
            print(f"Error in iteration {i}: {e}")
            print("Skipping to next iteration...")
            traceback.print_exc()

            # to deal with possible untocked timer counts
            timer.force_tock_everything()

            continue
        except ValueError as e:
            # sometimes wandb will log NaNs
            print(update_info)
            raise e

if __name__ == "__main__":
    app.run(main)
