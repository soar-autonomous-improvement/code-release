import os
from typing import Any
import traceback
import jax
import numpy as np
import orbax.checkpoint
from jaxrl_m.agents import agents
from jaxrl_m.data.bridge_dataset import BridgeDataset
from jaxrl_m.vision import encoders
import random
import time


class BasePolicy:
    def __init__(self, config):
        pass
    
    def get_update_step(self):
        return self.agent.state.step


class GCBCPolicy(BasePolicy):
    def __init__(self, config):
        self.gc_config = config["gc_policy_params"]
        train_paths = [[
            os.path.join(self.gc_config["mini_dataset_path"], "0.tfrecord")
        ]]
        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": self.gc_config["ACT_MEAN"],
                "std": self.gc_config["ACT_STD"],
                "min": self.gc_config["ACT_MEAN"], # we don't use this value
                "max": self.gc_config["ACT_STD"], # we don't use this value
            },
            "proprio": {
                "mean": self.gc_config["ACT_MEAN"], # we don't use this value
                "std": self.gc_config["ACT_STD"], # we don't use this value
                "min": self.gc_config["ACT_MEAN"], # we don't use this value
                "max": self.gc_config["ACT_STD"] # we don't use this value
            },
        }
        self.action_statistics = {"mean": self.gc_config["ACT_MEAN"], "std": self.gc_config["ACT_STD"]}
        train_data = BridgeDataset(
            train_paths,
            42,
            batch_size=256,
            num_devices=1,
            train=True,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            sample_weights=None,
            paths_index_with_old_tf_format=[],
            **self.gc_config["dataset_kwargs"],
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        encoder_config = self.gc_config["encoder"]
        encoder_def = encoders[encoder_config["type"]](**encoder_config["config"])
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        self.agent = agents[self.gc_config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **self.gc_config["agent_kwargs"],
        )

        self.update_weights()

        self.image_size = self.gc_config["image_size"]

        # Sticky gripper
        self.sticky_gripper_num_steps = config["general_params"]["sticky_gripper_num_steps"]
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

        # Workspace bounds
        self.bounds_x = config["general_params"]["manual_workspace_bounds"]["x"]
        self.bounds_y = config["general_params"]["manual_workspace_bounds"]["y"]
        self.bounds_z = config["general_params"]["manual_workspace_bounds"]["z"]
        self.min_xyz = np.array([self.bounds_x[0], self.bounds_y[0], self.bounds_z[0]])
        self.max_xyz = np.array([self.bounds_x[1], self.bounds_y[1], self.bounds_z[1]])

        # Exploration noise
        self.exploration = self.gc_config["exploration"]

    def update_weights(self):
        while True:
            try:
                print("Updating low-level policy checkpoint...")
                resume_path = self.gc_config["checkpoint_path"]
                restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=self.agent)
                if self.agent is restored:
                    raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
                print("Checkpoint successfully loaded")
                self.agent = restored
                break
            except:
                raise
                print("Error loading checkpoint, retrying...")

    def reset(self):
        """
            Reset is called when the task changes.
        """
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

    def __call__(self, obs_image: np.ndarray, goal_image: np.ndarray, pose: np.ndarray, deterministic=True):
        assert obs_image.shape == (self.image_size, self.image_size, 3), "Bad input obs image shape"
        assert goal_image.shape == (self.image_size, self.image_size, 3), "Bad input goal image shape"

        temperature = self.exploration["sampling_temperature"] if not deterministic else 0.0
        action_mode = self.agent.sample_actions(
            {"image" : obs_image[np.newaxis, ...]},
            {"image" : goal_image[np.newaxis, ...]}, 
            temperature=temperature,
            argmax=True,
        )
        action, action_mode = self.agent.sample_actions(
            {"image" : obs_image[np.newaxis, ...]},
            {"image" : goal_image[np.newaxis, ...]}, 
            temperature=temperature,
            argmax=deterministic,
            seed=None if deterministic else jax.random.PRNGKey(int(time.time())),
        )
        action, action_mode = np.array(action.tolist()), np.array(action_mode.tolist())
        action, action_mode = action[0], action_mode[0]

        # Remove exploration in unwanted dimensions
        action[3] = action_mode[3] # yaw
        action[4] = action_mode[4] # pitch
        action[-1] = action_mode[-1] # gripper

        print("Commanded gripper action:", action[-1].item())

        # Scale action
        action[:6] = np.array(self.action_statistics["std"][:6]) * action[:6] + np.array(self.action_statistics["mean"][:6])
        action_mode[:6] = np.array(self.action_statistics["std"][:6]) * action_mode[:6] + np.array(self.action_statistics["mean"][:6])

        # Sticky gripper logic
        if (action[-1] < 0.0) != (self.gripper_state == "closed"):
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.gripper_state == "closed" else 1.0

        # Add gripper noise
        if not deterministic:
            assert self.sticky_gripper_num_steps == 1
            switch_gripper_action_threshold = self.exploration["gripper_open_prob"] if action[-1] == 0.0 else self.exploration["gripper_close_prob"]
            if random.random() < switch_gripper_action_threshold:
                action[-1] = 0.0 if action[-1] == 1.0 else 1.0
        
        if self.gc_config["open_gripper_if_nothing_grasped"]:
            # If the gripper is completely closed, that means the grasp was unsuccessful. In that case, let's open the gripper
            if pose[-1] < 0.15:
                action[-1] = 1.0

        if self.gc_config["restrict_action_space"]:
            # Turn off pitch and yaw dimensions of gripper action
            action[4] = -0.1 - pose[4] # reset dimension to known optimal (zero) value
            action[3] = -pose[3]

        # Clip action to satisfy workspace bounds
        min_action = self.min_xyz - pose[:3]
        max_action = self.max_xyz - pose[:3]
        action[:3] = np.clip(action[:3], min_action, max_action)

        return action

class DiffusionPolicy(BasePolicy):
    def __init__(self, config):
        gc_config = config["gc_policy_params"]
        self.gc_config = gc_config
        train_paths = [[os.path.join(gc_config["mini_dataset_path"], "0.tfrecord")]]
        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": gc_config["ACT_MEAN"],
                "std": gc_config["ACT_STD"],
                "min": gc_config["ACT_MEAN"],  # we don't use this value
                "max": gc_config["ACT_STD"],  # we don't use this value
            },
            "proprio": {
                "mean": gc_config["PROPRIO_MEAN"],
                "std": gc_config["PROPRIO_STD"],
                "min": gc_config["ACT_MEAN"],  # we don't use this value
                "max": gc_config["ACT_STD"],  # we don't use this value
            },
        }
        self.action_statistics = {
            "mean": gc_config["ACT_MEAN"],
            "std": gc_config["ACT_STD"],
        }
        self.proprio_statistics = {
            "mean": np.array(gc_config["PROPRIO_MEAN"]),
            "std": np.array(gc_config["PROPRIO_STD"])
        }
        train_data = BridgeDataset(
            train_paths,
            42,
            batch_size=256,
            num_devices=1,
            train=True,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            sample_weights=None,
            **gc_config["dataset_kwargs"],
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        encoder_config = gc_config["encoder"]
        encoder_def = encoders[encoder_config["type"]](**encoder_config["config"])
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        self.agent = agents[gc_config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **gc_config["agent_kwargs"],
        )

        self.update_weights()

        self.image_size = gc_config["image_size"]

        # Prepare action buffer for temporal ensembling
        act_pred_horizon = gc_config["dataset_kwargs"]["act_pred_horizon"]
        self.action_buffer = np.zeros(
            (act_pred_horizon, act_pred_horizon, 7)
        )  # no need to make action dimension a config param
        self.action_buffer_mask = np.zeros(
            (act_pred_horizon, act_pred_horizon), dtype=bool
        )
        self.action_buffer_mask_mask = np.zeros(
            (act_pred_horizon, act_pred_horizon), dtype=bool
        )
        for i in range(act_pred_horizon):
            self.action_buffer_mask_mask[i, : act_pred_horizon - i] = True

        # Sticky gripper
        self.sticky_gripper_num_steps = config["general_params"]["sticky_gripper_num_steps"]
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

        # Exploration noise
        self.sigma = gc_config["exploration_noise"]

    def update_weights(self):
        while True:
            try:
                print("Updating low-level policy checkpoint...")
                resume_path = self.gc_config["checkpoint_path"]
                restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=self.agent)
                if self.agent is restored:
                    raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
                print("Checkpoint successfully loaded")
                self.agent = restored
                break
            except:
                print("Error loading checkpoint, retrying...")

    def reset(self):
        """
        Reset is called when the task changes.
        """
        self.action_buffer = np.zeros(self.action_buffer.shape)
        self.action_buffer_mask = np.zeros(self.action_buffer_mask.shape, dtype=bool)
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

    def __call__(self, obs_image: np.ndarray, goal_image: np.ndarray, pose: np.ndarray, deterministic=False):
        assert obs_image.shape == (
            self.image_size,
            self.image_size,
            3,
        ), "Bad input obs image shape"
        assert goal_image.shape == (
            self.image_size,
            self.image_size,
            3,
        ), "Bad input goal image shape"

        action = self.agent.sample_actions(
            {"image": obs_image[np.newaxis, ...], "proprio": (pose - self.proprio_statistics["mean"]) / self.proprio_statistics["std"]},
            {"image": goal_image},
            seed=jax.random.PRNGKey(42),
            temperature=0.0,
        )
        action = np.array(action.tolist())

        # Scale action
        action = np.array(self.action_statistics["std"]) * action + np.array(
            self.action_statistics["mean"]
        )
        print(action[0])

        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.action_buffer_mask_mask

        # Add to action buffer
        self.action_buffer[0] = action
        self.action_buffer_mask[0] = np.array(
            [True] * len(self.action_buffer), dtype=bool
        )

        # Ensemble temporally to predict action
        action = np.sum(
            self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0
        ) / np.sum(self.action_buffer_mask[:, 0], axis=0)

        print(action)
        print()

        # Sticky gripper logic
        if (action[-1] < 0.0) != (self.gripper_state == "closed"):
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.gripper_state == "closed" else 1.0

        # Add a little bit of Gaussian noise to first six dimensions
        # for exploration
        action[:-1] = action[:-1] + np.random.normal(0.0, self.sigma, (6,))

        return action

class RCSLPolicy(BasePolicy):
    def __init__(self, config):
        gc_config = config["gc_policy_params"]
        train_paths = [[os.path.join(gc_config["mini_dataset_path"], "0.tfrecord")]]
        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": gc_config["ACT_MEAN"],
                "std": gc_config["ACT_STD"],
                "min": gc_config["ACT_MEAN"],  # we don't use this value
                "max": gc_config["ACT_STD"],  # we don't use this value
            },
            "proprio": {
                "mean": gc_config["ACT_MEAN"],  # we don't use this value
                "std": gc_config["ACT_STD"],  # we don't use this value
                "min": gc_config["ACT_MEAN"],  # we don't use this value
                "max": gc_config["ACT_STD"],  # we don't use this value
            },
        }
        self.action_statistics = {
            "mean": gc_config["ACT_MEAN"],
            "std": gc_config["ACT_STD"],
        }
        train_data = BridgeDataset(
            train_paths,
            42,
            batch_size=256,
            num_devices=1,
            train=True,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            sample_weights=None,
            **gc_config["dataset_kwargs"],
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        encoder_config = gc_config["encoder"]
        encoder_def = encoders[encoder_config["type"]](**encoder_config["config"])
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        self.agent = agents[gc_config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **gc_config["agent_kwargs"],
        )

        print("Loading low-level policy checkpoint...")
        resume_path = gc_config["checkpoint_path"]
        restored = orbax.checkpoint.PyTreeCheckpointer().restore(
            resume_path, item=self.agent
        )
        if self.agent is restored:
            raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
        print("Checkpoint successfully loaded")
        self.agent = restored

        self.image_size = gc_config["image_size"]

        # Prepare action buffer for temporal ensembling
        act_pred_horizon = gc_config["dataset_kwargs"]["act_pred_horizon"]
        self.action_buffer = np.zeros(
            (act_pred_horizon, act_pred_horizon, 7)
        )  # no need to make action dimension a config param
        self.action_buffer_mask = np.zeros(
            (act_pred_horizon, act_pred_horizon), dtype=bool
        )
        self.action_buffer_mask_mask = np.zeros(
            (act_pred_horizon, act_pred_horizon), dtype=bool
        )
        for i in range(act_pred_horizon):
            self.action_buffer_mask_mask[i, : act_pred_horizon - i] = True

        # Sticky gripper
        self.sticky_gripper_num_steps = config["general_params"]["sticky_gripper_num_steps"]
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

        # Exploration noise
        self.sigma = gc_config["exploration_noise"]

    def reset(self):
        """
        Reset is called when the task changes.
        """
        self.action_buffer = np.zeros(self.action_buffer.shape)
        self.action_buffer_mask = np.zeros(self.action_buffer_mask.shape, dtype=bool)
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

    def __call__(self, obs_image: np.ndarray, goal_image: np.ndarray, pose: np.ndarray):
        assert obs_image.shape == (
            self.image_size,
            self.image_size,
            3,
        ), "Bad input obs image shape"
        assert goal_image.shape == (
            self.image_size,
            self.image_size,
            3,
        ), "Bad input goal image shape"

        action = self.agent.sample_actions(
            {"image": np.repeat(obs_image[np.newaxis, ...], 256, axis=0)},
            {"image": np.repeat(goal_image[np.newaxis, ...], 256, axis=0)},
            np.array([-10.0] * 256),
            seed=jax.random.PRNGKey(42),
            temperature=0.0,
        )
        action = np.array(action.tolist())[0]

        # Scale action
        action = np.array(self.action_statistics["std"]) * action + np.array(
            self.action_statistics["mean"]
        )

        # Shift action buffer
        self.action_buffer[1:, :, :] = self.action_buffer[:-1, :, :]
        self.action_buffer_mask[1:, :] = self.action_buffer_mask[:-1, :]
        self.action_buffer[:, :-1, :] = self.action_buffer[:, 1:, :]
        self.action_buffer_mask[:, :-1] = self.action_buffer_mask[:, 1:]
        self.action_buffer_mask = self.action_buffer_mask * self.action_buffer_mask_mask

        # Add to action buffer
        self.action_buffer[0] = action
        self.action_buffer_mask[0] = np.array(
            [True] * len(self.action_buffer), dtype=bool
        )

        # Ensemble temporally to predict action
        action = np.sum(
            self.action_buffer[:, 0, :] * self.action_buffer_mask[:, 0:1], axis=0
        ) / np.sum(self.action_buffer_mask[:, 0], axis=0)

        # Sticky gripper logic
        if (action[-1] < 0.5) != (self.gripper_state == "closed"):
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.gripper_state == "closed" else 1.0

        # Add a little bit of Gaussian noise to first six dimensions
        # for exploration
        action[:-1] = action[:-1] + np.random.normal(0.0, self.sigma, (6,))

        return action

class ContrastiveRLPolicy(BasePolicy):
    def __init__(self, config):
        self.gc_config = config["gc_policy_params"]
        train_paths = [[
            os.path.join(self.gc_config["mini_dataset_path"], "0.tfrecord")
        ]]
        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": self.gc_config["ACT_MEAN"],
                "std": self.gc_config["ACT_STD"],
                "min": self.gc_config["ACT_MEAN"], # we don't use this value
                "max": self.gc_config["ACT_STD"], # we don't use this value
            },
            "proprio": {
                "mean": self.gc_config["ACT_MEAN"], # we don't use this value
                "std": self.gc_config["ACT_STD"], # we don't use this value
                "min": self.gc_config["ACT_MEAN"], # we don't use this value
                "max": self.gc_config["ACT_STD"] # we don't use this value
            },
        }
        self.action_statistics = {"mean": self.gc_config["ACT_MEAN"], "std": self.gc_config["ACT_STD"]}
        train_data = BridgeDataset(
            train_paths,
            42,
            batch_size=256,
            num_devices=1,
            train=True,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            sample_weights=None,
            **self.gc_config["dataset_kwargs"],
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        encoder_config = self.gc_config["encoder"]
        encoder_def = encoders[encoder_config["type"]](**encoder_config["config"])
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        self.agent = agents[self.gc_config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **self.gc_config["agent_kwargs"],
        )

        self.update_weights(should_retry=False)

        self.image_size = self.gc_config["image_size"]

        # Sticky gripper
        self.sticky_gripper_num_steps = config["general_params"]["sticky_gripper_num_steps"]
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

        # Workspace bounds
        self.bounds_x = config["general_params"]["manual_workspace_bounds"]["x"]
        self.bounds_y = config["general_params"]["manual_workspace_bounds"]["y"]
        self.bounds_z = config["general_params"]["manual_workspace_bounds"]["z"]
        self.min_xyz = np.array([self.bounds_x[0], self.bounds_y[0], self.bounds_z[0]])
        self.max_xyz = np.array([self.bounds_x[1], self.bounds_y[1], self.bounds_z[1]])

        # Exploration noise
        self.exploration = self.gc_config["exploration"]

        # Sample and rank
        self.sample_and_rank = self.gc_config["sample_and_rank"]["activated"]
        if self.sample_and_rank:
            self.num_samples = self.gc_config["sample_and_rank"]["num_samples"]

    def update_weights(self, should_retry: bool = True):
        while True:
            try:
                print("Updating low-level policy checkpoint...")
                resume_path = self.gc_config["checkpoint_path"]
                restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=self.agent)
                if self.agent is restored:
                    raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
                print("Checkpoint successfully loaded")
                self.agent = restored
                break
            except:
                print("Error loading checkpoint, retrying...")
                if not should_retry:
                    exit()

    def reset(self):
        """
            Reset is called when the task changes.
        """
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

    def __call__(self, obs_image: np.ndarray, goal_image: np.ndarray, pose: np.ndarray, deterministic=False):
        assert obs_image.shape == (self.image_size, self.image_size, 3), "Bad input obs image shape"
        assert goal_image.shape == (self.image_size, self.image_size, 3), "Bad input goal image shape"

        if self.sample_and_rank:
            observations_dict = {
                "image" : np.tile(obs_image[np.newaxis, ...], (self.num_samples, 1, 1, 1)),
                "proprio" : np.tile(pose[np.newaxis, ...], (self.num_samples, 1)),
            }
            goals_dict = {
                "image" : np.tile(goal_image[np.newaxis, ...], (self.num_samples, 1, 1, 1)),
            }
            sample_and_rank_action, sample_and_rank_action_mode = self.agent.sample_actions(
                observations_dict,
                goals_dict, 
                seed=jax.random.PRNGKey(int(time.time())), 
                temperature=self.exploration["sampling_temperature"], 
                argmax=False,
            )
            # Concatenate the argmax action before evaluation so we can determine its value
            observations_dict["image"] = np.concatenate([observations_dict["image"], observations_dict["image"][:1]], axis=0)
            observations_dict["proprio"] = np.concatenate([observations_dict["proprio"], observations_dict["proprio"][:1]], axis=0)
            goals_dict["image"] = np.concatenate([goals_dict["image"], goals_dict["image"][:1]], axis=0)
            sampled_and_argmax_actions = np.concatenate([sample_and_rank_action, sample_and_rank_action_mode[:1]], axis=0)
            values = self.agent.rank_actions(
                observations_dict,
                goals_dict,
                sampled_and_argmax_actions,
            )
            values = np.array(values.tolist())
            assert len(values.shape) == 1
            
            action, action_mode = np.array(sample_and_rank_action.tolist()), np.array(sample_and_rank_action_mode.tolist())
            argmax_idx = np.argmax(values[:-1])
            print("Sample max value:", values[argmax_idx], "\tArgmax value:", values[-1])
            action, action_mode = action[argmax_idx], action_mode[0]

        else:
            temperature = self.exploration["sampling_temperature"] if not deterministic else 0.0
            action, action_mode = self.agent.sample_actions(
                {"image" : obs_image[np.newaxis, ...], "proprio": pose[np.newaxis, ...]},
                {"image" : goal_image[np.newaxis, ...]}, 
                seed=jax.random.PRNGKey(int(time.time())), 
                temperature=temperature, 
                argmax=deterministic,
            )
            action, action_mode = np.array(action.tolist()), np.array(action_mode.tolist())
            action, action_mode = action[0], action_mode[0]

        # Scale action
        action = np.array(self.action_statistics["std"]) * action + np.array(self.action_statistics["mean"])
        action_mode = np.array(self.action_statistics["std"]) * action_mode + np.array(self.action_statistics["mean"])

        # Remove exploration in unwanted dimensions
        action[3] = action_mode[3] # yaw
        action[4] = action_mode[4] # pitch
        action[-1] = action_mode[-1] # gripper

        print("Commanded gripper action:", action[-1].item())

        # Sticky gripper logic
        if (action[-1] < 0.0) != (self.gripper_state == "closed"):
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.gripper_state == "closed" else 1.0

        # Add gripper noise
        if not deterministic:
            assert self.sticky_gripper_num_steps == 1
            switch_gripper_action_threshold = self.exploration["gripper_open_prob"] if action[-1] == 0.0 else self.exploration["gripper_close_prob"]
            if random.random() < switch_gripper_action_threshold:
                action[-1] = 0.0 if action[-1] == 1.0 else 1.0
        
        if self.gc_config["open_gripper_if_nothing_grasped"]:
            # If the gripper is completely closed, that means the grasp was unsuccessful. In that case, let's open the gripper
            if pose[-1] < 0.15:
                action[-1] = 1.0

        if self.gc_config["restrict_action_space"]:
            # Turn off pitch and yaw dimensions of gripper action
            action[4] = -0.1 - pose[4] # reset dimension to known optimal (zero) value
            action[3] = -pose[3]

        # Clip action to satisfy workspace bounds
        min_action = self.min_xyz - pose[:3]
        max_action = self.max_xyz - pose[:3]
        action[:3] = np.clip(action[:3], min_action, max_action)

        return action

class CalQLPolicy(BasePolicy):
    def __init__(self, config):
        self.gc_config = config["gc_policy_params"]
        train_paths = [[
            os.path.join(self.gc_config["mini_dataset_path"], "0.tfrecord")
        ]]
        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": np.array(self.gc_config["ACT_MEAN"]),
                "std": np.array(self.gc_config["ACT_STD"]),
                "min": np.array(self.gc_config["ACT_MIN"]),
                "max": np.array(self.gc_config["ACT_MAX"]),
            },
            "proprio": {
                "mean": np.array(self.gc_config["ACT_MEAN"]), # we don't use this value
                "std": np.array(self.gc_config["ACT_MEAN"]), # we don't use this value
                "min": np.array(self.gc_config["ACT_MEAN"]), # we don't use this value
                "max": np.array(self.gc_config["ACT_MEAN"]), # we don't use this value
            },
        }
        self.action_statistics = ACTION_PROPRIO_METADATA["action"]
        train_data = BridgeDataset(
            train_paths,
            42,
            batch_size=256,
            num_devices=1,
            train=True,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            sample_weights=None,
            paths_index_with_old_tf_format=[0],
            **self.gc_config["dataset_kwargs"],
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        encoder_config = self.gc_config["encoder"]
        encoder_def = encoders[encoder_config["type"]](**encoder_config["config"])
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        self.agent = agents[self.gc_config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **self.gc_config["agent_kwargs"],
        )

        self.update_weights(should_retry=False)

        self.image_size = self.gc_config["image_size"]

        # Sticky gripper
        self.sticky_gripper_num_steps = config["general_params"]["sticky_gripper_num_steps"]
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

        # Workspace bounds
        self.bounds_x = config["general_params"]["manual_workspace_bounds"]["x"]
        self.bounds_y = config["general_params"]["manual_workspace_bounds"]["y"]
        self.bounds_z = config["general_params"]["manual_workspace_bounds"]["z"]
        self.min_xyz = np.array([self.bounds_x[0], self.bounds_y[0], self.bounds_z[0]])
        self.max_xyz = np.array([self.bounds_x[1], self.bounds_y[1], self.bounds_z[1]])

        # Exploration noise
        self.exploration = self.gc_config["exploration"]

    def update_weights(self, should_retry: bool = True):
        while True:
            try:
                print("Updating low-level policy checkpoint...")
                resume_path = self.gc_config["checkpoint_path"]
                restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=self.agent)
                if self.agent is restored:
                    raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
                print("Checkpoint successfully loaded")
                self.agent = restored
                break
            except:
                print("Error loading checkpoint, retrying...")
                if not should_retry:
                    exit()

    def reset(self):
        """
            Reset is called when the task changes.
        """
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

    def __call__(self, obs_image: np.ndarray, goal_image: np.ndarray, pose: np.ndarray, deterministic=False):
        assert obs_image.shape == (self.image_size, self.image_size, 3), "Bad input obs image shape"
        assert goal_image.shape == (self.image_size, self.image_size, 3), "Bad input goal image shape"

        temperature = self.exploration["sampling_temperature"] if not deterministic else 0.0
        action = self.agent.sample_actions(
                                {"image" : obs_image[np.newaxis, ...]},
                                {"image" : goal_image[np.newaxis, ...]}, 
                                argmax=deterministic,
                                temperature=temperature, 
                            )
        
        # values = self.agent.forward_critic(
        #     ({"image" : np.repeat(obs_image[np.newaxis, ...], 10, axis=0)}, {"image" : np.repeat(goal_image[np.newaxis, ...], 10, axis=0)}),
        #     actions, 
        #     rng=None,
        #     train=False
        # )
        # best_action_idx = jax.numpy.argmax(jax.numpy.min(values, 1))
        # action = actions[best_action_idx]
        
        action = np.array(action.tolist())
        action = action[0]

        # unnormalize action
        normalization_type = self.gc_config["dataset_kwargs"]["normalization_type"]
        if normalization_type == "tanh_normal":
            action[:6] = 4 * self.action_statistics["std"][:6] * action[:6] + self.action_statistics["mean"][:6]
        elif normalization_type == "tanh":
            action[:6] = (self.action_statistics["max"] - self.action_statistics["min"])[:6] * (action[:6] + 1) / 2 + self.action_statistics["min"][:6]
        else:
            raise ValueError(f"Unknown normalization type: {normalization_type}")

        # Sticky gripper logic
        convert_to_gripper_close = False
        if (action[-1] < 0.0) != (self.gripper_state == "closed"):
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            convert_to_gripper_close = self.gripper_state == "open"
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.gripper_state == "closed" else 1.0

        # Add gripper noise
        if not deterministic:
            assert self.sticky_gripper_num_steps == 1
            switch_gripper_action_threshold = self.exploration["gripper_open_prob"] if action[-1] == 0.0 else self.exploration["gripper_close_prob"]
            if random.random() < switch_gripper_action_threshold:
                action[-1] = 0.0 if action[-1] == 1.0 else 1.0

        if self.gc_config["clip_action"]:
            action[:6] = np.clip(action[:6], np.array(self.gc_config["clip_lower"]), np.array(self.gc_config["clip_upper"]))
        
        if self.gc_config["open_gripper_if_nothing_grasped"]:
            # If the gripper is completely closed, that means the grasp was unsuccessful. In that case, let's open the gripper
            if pose[-1] < 0.15:
                action[-1] = 1.0

        if self.gc_config["restrict_action_space"]:
            # Turn off pitch and yaw dimensions of gripper action
            action[4] = -0.1 - pose[4] # reset dimension to known optimal (zero) value
            action[3] = -pose[3]

        # Clip action to satisfy workspace bounds
        min_action = self.min_xyz - pose[:3]
        max_action = self.max_xyz - pose[:3]
        action[:3] = np.clip(action[:3], min_action, max_action)

        return action
    
class IQLPolicy(BasePolicy):
    def __init__(self, config):
        self.gc_config = config["gc_policy_params"]
        train_paths = [[
            os.path.join(self.gc_config["mini_dataset_path"], "0.tfrecord")
        ]]
        ACTION_PROPRIO_METADATA = {
            "action": {
                "mean": self.gc_config["ACT_MEAN"],
                "std": self.gc_config["ACT_STD"],
                "min": self.gc_config["ACT_MEAN"], # we don't use this value
                "max": self.gc_config["ACT_STD"], # we don't use this value
            },
            "proprio": {
                "mean": self.gc_config["ACT_MEAN"], # we don't use this value
                "std": self.gc_config["ACT_STD"], # we don't use this value
                "min": self.gc_config["ACT_MEAN"], # we don't use this value
                "max": self.gc_config["ACT_STD"] # we don't use this value
            },
        }
        self.action_statistics = {"mean": self.gc_config["ACT_MEAN"], "std": self.gc_config["ACT_STD"]}
        train_data = BridgeDataset(
            train_paths,
            42,
            batch_size=256,
            num_devices=1,
            train=True,
            action_proprio_metadata=ACTION_PROPRIO_METADATA,
            sample_weights=None,
            paths_index_with_old_tf_format=[0],
            **self.gc_config["dataset_kwargs"],
        )
        train_data_iter = train_data.iterator()
        example_batch = next(train_data_iter)

        encoder_config = self.gc_config["encoder"]
        encoder_def = encoders[encoder_config["type"]](**encoder_config["config"])
        rng = jax.random.PRNGKey(42)
        rng, construct_rng = jax.random.split(rng)
        self.agent = agents[self.gc_config["policy_class"]].create(
            rng=construct_rng,
            observations=example_batch["observations"],
            goals=example_batch["goals"],
            actions=example_batch["actions"],
            encoder_def=encoder_def,
            **self.gc_config["agent_kwargs"],
        )

        self.update_weights(should_retry=False)

        self.image_size = self.gc_config["image_size"]

        # Sticky gripper
        self.sticky_gripper_num_steps = config["general_params"]["sticky_gripper_num_steps"]
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

        # Workspace bounds
        self.bounds_x = config["general_params"]["manual_workspace_bounds"]["x"]
        self.bounds_y = config["general_params"]["manual_workspace_bounds"]["y"]
        self.bounds_z = config["general_params"]["manual_workspace_bounds"]["z"]
        self.min_xyz = np.array([self.bounds_x[0], self.bounds_y[0], self.bounds_z[0]])
        self.max_xyz = np.array([self.bounds_x[1], self.bounds_y[1], self.bounds_z[1]])

        # Exploration noise
        self.exploration = self.gc_config["exploration"]

    def update_weights(self, should_retry: bool = True):
        while True:
            try:
                print("Updating low-level policy checkpoint...")
                resume_path = self.gc_config["checkpoint_path"]
                restored = orbax.checkpoint.PyTreeCheckpointer().restore(resume_path, item=self.agent)
                if self.agent is restored:
                    raise FileNotFoundError(f"Cannot load checkpoint from {resume_path}")
                print("Checkpoint successfully loaded")
                self.agent = restored
                break
            except:
                print("Error loading checkpoint, retrying...")
                raise
                if not should_retry:
                    exit()

    def reset(self):
        """
            Reset is called when the task changes.
        """
        self.num_consecutive_gripper_change_actions = 0
        self.gripper_state = "open"

    def __call__(self, obs_image: np.ndarray, goal_image: np.ndarray, pose: np.ndarray, deterministic=False):
        assert obs_image.shape == (self.image_size, self.image_size, 3), "Bad input obs image shape"
        assert goal_image.shape == (self.image_size, self.image_size, 3), "Bad input goal image shape"

        temperature = self.exploration["sampling_temperature"] if not deterministic else 0.0
        action, action_mode = self.agent.sample_actions(
                                {"image" : obs_image[np.newaxis, ...]},
                                {"image" : goal_image[np.newaxis, ...]}, 
                                temperature=temperature,
                                argmax=deterministic,
                                seed=jax.random.PRNGKey(int(time.time())),
                            )
        action, action_mode = np.array(action.tolist()), np.array(action_mode.tolist())
        action, action_mode = action[0], action_mode[0]

        # Scale action
        action[:6] = np.array(self.action_statistics["std"][:6]) * action[:6] + np.array(self.action_statistics["mean"][:6])
        action_mode[:6] = np.array(self.action_statistics["std"][:6]) * action_mode[:6] + np.array(self.action_statistics["mean"][:6])

        # Remove exploration in unwanted dimensions
        action[3] = action_mode[3] # yaw
        action[4] = action_mode[4] # pitch
        action[-1] = action_mode[-1] # gripper

        print("Commanded gripper action:", action[-1].item())

        # Sticky gripper logic
        if (action[-1] < 0.0) != (self.gripper_state == "closed"):
            self.num_consecutive_gripper_change_actions += 1
        else:
            self.num_consecutive_gripper_change_actions = 0

        if self.num_consecutive_gripper_change_actions >= self.sticky_gripper_num_steps:
            self.gripper_state = "open" if self.gripper_state == "closed" else "closed"
            self.num_consecutive_gripper_change_actions = 0
        action[-1] = 0.0 if self.gripper_state == "closed" else 1.0

        # Add gripper noise
        if not deterministic:
            assert self.sticky_gripper_num_steps == 1
            switch_gripper_action_threshold = self.exploration["gripper_open_prob"] if action[-1] == 0.0 else self.exploration["gripper_close_prob"]
            if random.random() < switch_gripper_action_threshold:
                action[-1] = 0.0 if action[-1] == 1.0 else 1.0
        
        if self.gc_config["open_gripper_if_nothing_grasped"]:
            # If the gripper is completely closed, that means the grasp was unsuccessful. In that case, let's open the gripper
            if pose[-1] < 0.15:
                action[-1] = 1.0

        if self.gc_config["restrict_action_space"]:
            # Turn off pitch and yaw dimensions of gripper action
            action[4] = -0.1 - pose[4] # reset dimension to known optimal (zero) value
            action[3] = -pose[3]

        # Clip action to satisfy workspace bounds
        min_action = self.min_xyz - pose[:3]
        max_action = self.max_xyz - pose[:3]
        action[:3] = np.clip(action[:3], min_action, max_action)

        return action


gc_policies = {
    "gc_bc": GCBCPolicy,
    "gc_ddpm_bc": DiffusionPolicy,
    "stable_contrastive_rl": ContrastiveRLPolicy,
    "calql": CalQLPolicy,
    "gc_iql": IQLPolicy,
    "cql": CalQLPolicy,
}
