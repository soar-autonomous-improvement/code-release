"""
tf.data.Dataset based dataloader for the BridgeData format, meaning
TFRecords with one trajectory per example. See the BridgeDataset class
below for more details.
"""

import fnmatch
from functools import partial
from typing import Iterable, List, Optional, Union

import numpy as np
import tensorflow as tf
from absl import logging

from jaxrl_m.data.tf_augmentations import augment
from jaxrl_m.data.tf_goal_relabeling import GOAL_RELABELING_FUNCTIONS


def glob_to_path_list(
    glob_strs: Union[str, List[str]], prefix: str = "", exclude: Iterable[str] = ()
):
    """Converts a glob string or list of glob strings to a list of paths."""
    if isinstance(glob_strs, str):
        glob_strs = [glob_strs]
    path_list = []
    for glob_str in glob_strs:
        paths = tf.io.gfile.glob(f"{prefix}/{glob_str}")
        filtered_paths = []
        for path in paths:
            if not any(fnmatch.fnmatch(path, e) for e in exclude):
                filtered_paths.append(path)
            else:
                logging.info(f"Excluding {path}")
        if len(filtered_paths) == 0:
            print("Warning: glob_to_path_list didn't find any paths")
        path_list += filtered_paths
    return path_list


@tf.function(jit_compile=True)
def _binarize_gripper_actions(actions):
    """Converts gripper actions from continous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near
    1.0) or fully closed (near 0.0). As it transitions between the two, it
    sometimes passes through a few intermediate values. We relabel those
    intermediate values based on the state that is reached _after_ those
    intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we
    give up on binarizing and relabel that chunk of intermediate values as
    the last action in the trajectory.

    The scan implements the following code:

    new_actions = np.empty_like(actions)
    carry = actions[-1]
    for i in reversed(range(actions.shape[0])):
        if in_between_mask[i]:
            carry = carry
        else:
            carry = float(open_mask[i])
        new_actions[i] = carry
    """
    open_mask = actions > 0.95
    closed_mask = actions < 0.05
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))

    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(
            in_between_mask[i],
            lambda: tf.cast(carry, tf.float32),
            lambda: is_open_float[i],
        )

    new_actions = tf.scan(
        scan_fn, tf.range(tf.shape(actions)[0]), actions[-1], reverse=True
    )

    return new_actions


def _binary_gripper_to_continuous(
    g_actions, gripper_action_mean=0.5, gripper_action_std=0.1
):
    # transform the actions from [0, 1] binary to [-1, 1] continuous
    is_open_mask = g_actions > 0.5
    is_open_mask = tf.cast(is_open_mask, tf.float32)
    traj_len = tf.shape(g_actions)[0]

    normal_samples = tf.random.normal(
        [traj_len], mean=gripper_action_mean, stddev=gripper_action_std
    )
    new_actions = normal_samples * is_open_mask + (
        normal_samples - gripper_action_mean * 2
    ) * (1 - is_open_mask)

    return new_actions


class BridgeDataset:
    """
    Fast parallel tf.data.Dataset-based dataloader for a dataset in the
    BridgeData format. This format consists of TFRecords where each example
    is one trajectory. See `PROTO_TYPE_SPEC` below for the expected format
    for each example in more detail. See `_process_trajectory` below for
    the output format.

    Includes goal relabeling, image augmentations, and sampling from multiple
    datasets with different weights. Goal relabeling uses a 0/-1 reward scheme:
    0 when the next_obs is labeled as the goal, -1 otherwise.

    Args:
        data_paths: List of paths to the data files. If a list of list of paths
            is provided, the data will be sampled from each sub-list according
            to "sample_weights".
        seed: Random seed.
        action_proprio_metadata: Dictionary containing metadata of the actions and proprio.
            If provided, actions and proprio will be normalized.
        normalization_type: The type of normalization to apply to the actions
            and proprio.
        action_clip_delta: If normalization bounds the agent to certain range, this
            clips the action to be within the bounds for another small delta value.
        relabel_actions: Whether to relabel the actions with reached states
            (based on proprioception). Also binarizes gripper actions.
        goal_relabeling_strategy: Goal relabeling strategy. See
            `jaxrl_m.data.tf_goal_relabeling` for more details.
        goal_relabeling_kwargs: Keyword arguments for goal relabeling. See
            `jaxrl_m.data.tf_goal_relabeling` for more details.
        sample_weights: If data_paths is a list of list of paths, this is a
            list of weights with which to sample from each sub-list.
        batch_size: Batch size.
        shuffle_buffer_size: Size of the shuffle buffer. It is split between
            sub-datasets by `sample_weights`.
        cache: Whether to cache the dataset in memory.
        train: Whether this dataset is intended for training
            (if set to `False`, will disable shuffling and augmentations).
        augment: Whether to apply image augmentations.
        augment_kwargs: Keyword arguments for image augmentations. See
            `jaxrl_m.data.tf_augmentations.augment` for more details.
        augment_next_obs_goal_differently: Whether to use different random seeds
            for augmenting the obs, next_obs, and goal image.
        act_pred_horizon: Number of consecutive actions that will be predicted.
        obs_horizon: Number of consecutive observations that will be conditioned on.
        load_langauge: Whether to look for and load language from the data.
        skip_unlabeled: Whether to filter out trajectories not labeled with language.
        action_merge_horizon: if > 1, sum actions over this many steps. if -1, use
            an adaptive scheme to automatically merge actions together until either
            a) the gripper action changes or b) the proprio sum exceeds the metadata threshold.
    """

    def __init__(
        self,
        data_paths: List[Union[str, List[str]]],
        seed: int,
        action_proprio_metadata: Optional[dict] = None,
        normalization_type: Optional[str] = "normal",
        action_clip_delta: float = 0,
        relabel_actions: bool = True,
        goal_relabeling_strategy: str = "uniform",
        goal_relabeling_kwargs: dict = {},
        sample_weights: Optional[List[float]] = None,
        batch_size: int = 256,
        shuffle_buffer_size: int = 10000,
        cache: bool = False,
        train: bool = True,
        augment: bool = False,
        augment_next_obs_goal_differently: bool = False,
        augment_kwargs: dict = {},
        act_pred_horizon: Optional[int] = None,
        obs_horizon: Optional[int] = None,
        load_language: bool = False,
        skip_unlabeled: bool = False,
        gripper_action_mean: float = 0.5,
        gripper_action_std: float = 0.1,
        dataset_contains_commanded_goals: bool = False,
        return_entire_trajectory: bool = False,
        paths_index_with_old_tf_format: Optional[List[int]] = [],
        action_merge_horizon: int = 1,
        static_eef_filter_threshold=0.1,
        **kwargs,
    ):
        logging.warning("Extra kwargs passed to BridgeDataset: %s", kwargs)
        if isinstance(data_paths[0], str):
            data_paths = [data_paths]
        if sample_weights is None:
            # default to uniform distribution over sub-lists
            sample_weights = [1 / len(data_paths)] * len(data_paths)
        assert len(data_paths) == len(sample_weights)
        assert np.isclose(sum(sample_weights), 1.0)

        self.relabel_actions = relabel_actions
        self.action_proprio_metadata = action_proprio_metadata
        self.normalization_type = normalization_type
        self.action_clip_delta = action_clip_delta
        self.goal_relabeling_strategy = goal_relabeling_strategy
        self.goal_relabeling_kwargs = goal_relabeling_kwargs
        self.cache = cache
        self.augment_kwargs = augment_kwargs
        self.augment_next_obs_goal_differently = augment_next_obs_goal_differently
        self.act_pred_horizon = act_pred_horizon
        self.obs_horizon = obs_horizon
        self.is_train = train
        self.load_language = load_language
        self.gripper_action_mean = gripper_action_mean
        self.gripper_action_std = gripper_action_std
        self.dataset_contains_commanded_goals = dataset_contains_commanded_goals
        self.return_entire_trajectory = return_entire_trajectory
        self.action_merge_horizon = action_merge_horizon
        self.static_eef_filter_threshold = static_eef_filter_threshold

        if self.load_language:
            self.PROTO_TYPE_SPEC["language"] = tf.string

        # construct a dataset for each sub-list of paths
        datasets = []
        diff_gc_kwargs_per_dataset = type(self.goal_relabeling_kwargs) == list
        if diff_gc_kwargs_per_dataset:
            logging.info("Multiple sets of goal relabeling kwargs passed in")
            all_goal_relabeling_kwargs = self.goal_relabeling_kwargs
        for i, sub_data_paths in enumerate(data_paths):
            if diff_gc_kwargs_per_dataset:
                self.goal_relabeling_kwargs = all_goal_relabeling_kwargs[i]
            datasets.append(
                self._construct_tf_dataset(
                    sub_data_paths,
                    seed,
                    use_old_tfrecord_format=i in paths_index_with_old_tf_format,
                )
            )

        if train:
            # shuffle and repeat each sub-dataset, allocating the shuffle buffer
            # by sample_weights
            for i in range(len(datasets)):
                datasets[i] = (
                    datasets[i]
                    .shuffle(int(shuffle_buffer_size * sample_weights[i]), seed + i)
                    .repeat()
                )

        # for validation, we want to be able to iterate through the entire dataset;
        # for training, we want to make sure that no sub-dataset is ever exhausted
        # or the sampling ratios will be off. this should never happen because of the
        # repeat() above, but `stop_on_empty_dataset` is a safeguard
        dataset = tf.data.Dataset.sample_from_datasets(
            datasets, sample_weights, seed=seed, stop_on_empty_dataset=train
        )

        if skip_unlabeled:
            dataset = dataset.filter(
                lambda x: tf.math.reduce_any(x["goals"]["language"] != "")
            )

        if train and augment:
            # apply augmentations, using a sequence of integers as seeds.
            # this was the only way I found to avoid a memory leak in tf.random.Generator
            dataset = dataset.enumerate(start=seed)
            dataset = dataset.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(
            batch_size,
            num_parallel_calls=tf.data.AUTOTUNE,
            drop_remainder=True,
            deterministic=not train,
        )

        self.tf_dataset = dataset

    def _construct_tf_dataset(
        self, paths: List[str], seed: int, use_old_tfrecord_format=False
    ) -> tf.data.Dataset:
        """
        Constructs a tf.data.Dataset from a list of paths.
        The dataset yields a dictionary of tensors for each transition.
        """

        # shuffle again using the dataset API so the files are read in a
        # different order every epoch
        dataset = tf.data.Dataset.from_tensor_slices(paths).shuffle(len(paths), seed)

        # yields raw serialized examples
        dataset = tf.data.TFRecordDataset(dataset, num_parallel_reads=tf.data.AUTOTUNE)

        # yields trajectories
        if use_old_tfrecord_format:
            # autonomous data
            dataset = dataset.map(
                self._decode_example_old_format, num_parallel_calls=tf.data.AUTOTUNE
            )

        else:
            # orca bridge data
            dataset = dataset.map(
                self._decode_example, num_parallel_calls=tf.data.AUTOTUNE
            )

        # yields trajectories
        dataset = dataset.map(
            partial(
                self._process_actions,
                merge_actions=self.action_merge_horizon != 1,
                crop_out_ends=False,  # only do it for autonomous data
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        # yields trajectories
        dataset = dataset.map(self._chunk_act_obs, num_parallel_calls=tf.data.AUTOTUNE)

        # cache before add_goals because add_goals introduces randomness
        if self.cache:
            dataset = dataset.cache()

        # yields trajectories
        dataset = dataset.map(self._add_goals, num_parallel_calls=tf.data.AUTOTUNE)

        # unbatch to yield individual transitions
        if not self.return_entire_trajectory:
            dataset = dataset.unbatch()

        return dataset

    # the expected type spec for the serialized examples
    PROTO_TYPE_SPEC = {
        "steps/observation/image_0": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.string, allow_missing=True
        ),  # Encoded images as a sequence
        "steps/observation/state": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),  # States as a sequence
        "steps/action": tf.io.FixedLenSequenceFeature(
            [], dtype=tf.float32, allow_missing=True
        ),  # Actions as a sequence
    }

    def _decode_example(self, serialized_example):
        # Parse the serialized example
        parsed_features = tf.io.parse_single_example(
            serialized_example, self.PROTO_TYPE_SPEC
        )

        # Decode images
        images_decoded = tf.map_fn(
            fn=lambda x: tf.io.decode_jpeg(x, channels=3),
            elems=parsed_features["steps/observation/image_0"],
            fn_output_signature=tf.TensorSpec(shape=[256, 256, 3], dtype=tf.uint8),
        )

        # No need to reshape images since we're already setting the shape in map_fn,
        # but we will have to reshape the states and actions
        states_reshaped = tf.reshape(
            parsed_features["steps/observation/state"], [-1, 7]
        )
        actions_reshaped = tf.reshape(parsed_features["steps/action"], [-1, 7])

        # restructure the dictionary into the downstream format
        return {
            "observations": {
                "image": images_decoded[:-1],
                "proprio": states_reshaped[:-1],
            },
            "next_observations": {
                "image": images_decoded[1:],
                "proprio": states_reshaped[1:],
            },
            "actions": actions_reshaped[:-1],
            "terminals": tf.zeros_like(actions_reshaped[:-1][:, 0]),
        }

    PROTO_TYPE_SPEC_OLD_FORMAT = {
        "observations/images0": tf.uint8,
        "observations/state": tf.float32,
        "next_observations/images0": tf.uint8,
        "next_observations/state": tf.float32,
        "actions": tf.float32,
        "terminals": tf.bool,
        "truncates": tf.bool,
    }

    def _decode_example_old_format(self, example_proto):
        if self.dataset_contains_commanded_goals:
            self.PROTO_TYPE_SPEC_OLD_FORMAT["commanded_goals"] = tf.uint8

        # decode the example proto according to PROTO_TYPE_SPEC
        features = {
            key: tf.io.FixedLenFeature([], tf.string)
            for key in self.PROTO_TYPE_SPEC_OLD_FORMAT.keys()
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)
        parsed_tensors = {
            key: tf.io.parse_tensor(parsed_features[key], dtype)
            for key, dtype in self.PROTO_TYPE_SPEC_OLD_FORMAT.items()
        }
        # restructure the dictionary into the downstream format
        return {
            "observations": {
                "image": parsed_tensors["observations/images0"],
                "proprio": parsed_tensors["observations/state"],
            },
            "next_observations": {
                "image": parsed_tensors["next_observations/images0"],
                "proprio": parsed_tensors["next_observations/state"],
            },
            "actions": parsed_tensors["actions"],
            "terminals": tf.zeros_like(parsed_tensors["actions"][:, 0]),
        }

    def _process_actions(self, traj, merge_actions=False, crop_out_ends=False):
        # relabel actions
        traj = self._relabel_actions(traj)

        # crop out low magnitude actions at the end of the trajectory
        if crop_out_ends:
            print(
                f"NOTE: Discarding low magnitude actions that occur at the end of autonomous trajectories"
            )
            traj = self._crop_out_ends(traj)

        # sum actions over a horizon
        if merge_actions:
            if self.action_merge_horizon > 1:
                print(f"NOTE: Summing actions over {self.action_merge_horizon} steps")
                traj = self._sum_actions(traj)
            else:
                assert self.action_merge_horizon == -1
                print(f"NOTE: Adaptive summing actions")
                traj = self._sum_actions_adaptive(traj)
        else:
            assert self.action_merge_horizon == 1

        # normalize actions and proprio
        traj = self._normalize_actions_proprio(traj)

        # distribute the discrete gripper action to a normal distribution
        new_gripper_actions = _binary_gripper_to_continuous(
            traj["actions"][:, 6],
            gripper_action_mean=self.gripper_action_mean,
            gripper_action_std=self.gripper_action_std,
        )
        traj["actions"] = tf.concat(
            [traj["actions"][:, :6], new_gripper_actions[:, None]], axis=1
        )

        return traj

    def _relabel_actions(self, traj):
        if self.relabel_actions:
            if self.action_merge_horizon == 1:
                # relabel the first 6 action dims (xyz position, xyz rotation)
                # using the reached proprio
                movement_actions = (
                    traj["next_observations"]["proprio"][:, :6]
                    - traj["observations"]["proprio"][:, :6]
                )
            else:
                assert self.action_merge_horizon > 1 or self.action_merge_horizon == -1
                # already extracted movement actions from proprio in _sum_actions()
                movement_actions = traj["actions"][:, :6]

            # binarize the gripper action
            gripper_actions = traj["actions"][:, 6]
            binarized_gripper_actions = _binarize_gripper_actions(
                gripper_actions,
            )

            traj["actions"] = tf.concat(
                [movement_actions, binarized_gripper_actions[:, None]],
                axis=1,
            )

        return traj

    def _normalize_actions_proprio(self, traj):
        # normalize actions and proprio
        if self.action_proprio_metadata is not None:
            if (
                self.normalization_type == "normal"
                or self.normalization_type == "tanh_normal"
            ):
                # normalize to mean 0, std 1
                traj["actions"] = tf.concat(
                    [
                        (
                            traj["actions"][:, :6]
                            - self.action_proprio_metadata["action"]["mean"][:6]
                        )
                        / self.action_proprio_metadata["action"]["std"][:6],
                        traj["actions"][:, 6:],
                    ],
                    axis=1,
                )
                if self.normalization_type == "tanh_normal":
                    traj["actions"] = tf.concat(
                        [traj["actions"][:, :6] / 4, traj["actions"][:, 6:]], axis=1
                    )  # makes prob of <-1 or >1 practically 0, as if we tanh'ed it
                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"]
                        - self.action_proprio_metadata["proprio"]["mean"]
                    ) / self.action_proprio_metadata["proprio"]["std"]
            elif self.normalization_type in ("bounds", "tanh"):
                # normalize to [0, 1]
                traj["actions"] = tf.concat(
                    [
                        (
                            traj["actions"][:, :6]
                            - self.action_proprio_metadata["action"]["min"][:6]
                        )
                        / (
                            self.action_proprio_metadata["action"]["max"][:6]
                            - self.action_proprio_metadata["action"]["min"][:6]
                        ),
                        traj["actions"][:, 6:],
                    ],
                    axis=1,
                )

                if self.normalization_type == "tanh":
                    # normalize to [-1, 1]
                    traj["actions"] = tf.concat(
                        [
                            traj["actions"][:, :6] * 2 - 1,
                            traj["actions"][:, 6:],
                        ],
                        axis=1,
                    )
                    traj["actions"] = tf.clip_by_value(
                        traj["actions"],
                        -1 + self.action_clip_delta,
                        1 - self.action_clip_delta,
                    )
                elif self.normalization_type == "bounds":
                    # clip to [0, 1]
                    traj["actions"] = tf.clip_by_value(
                        traj["actions"],
                        0 + self.action_clip_delta,
                        1 - self.action_clip_delta,
                    )

                for key in ["observations", "next_observations"]:
                    traj[key]["proprio"] = (
                        traj[key]["proprio"]
                        - self.action_proprio_metadata["proprio"]["min"]
                    ) / (
                        self.action_proprio_metadata["proprio"]["max"]
                        - self.action_proprio_metadata["proprio"]["min"]
                    )
                    if self.normalization_type == "tanh":
                        # normalize to [-1, 1]
                        traj[key]["proprio"] = traj[key]["proprio"] * 2 - 1
                        traj[key]["proprio"] = tf.clip_by_value(
                            traj[key]["proprio"],
                            -1 + self.action_clip_delta,
                            1 - self.action_clip_delta,
                        )
                    elif self.normalization_type == "bounds":
                        traj[key]["proprio"] = tf.clip_by_value(
                            traj[key]["proprio"],
                            0 + self.action_clip_delta,
                            1 - self.action_clip_delta,
                        )
            else:
                raise ValueError

        return traj

    def _crop_out_ends(self, traj):
        """
        We begin from the end of the trajectory and start summing action magnitudes. We keep
        cropping out actions until the sum of action magnitudes exceeds some threshold.
        """
        action_magnitudes = tf.math.sqrt(
            tf.math.reduce_sum(tf.math.square(traj["actions"][:, :6]), axis=1)
        )
        action_magnitudes_cumsum = tf.math.cumsum(action_magnitudes, reverse=True)
        cumsum_mask = action_magnitudes_cumsum > self.static_eef_filter_threshold
        cumsum_mask = tf.cast(cumsum_mask, tf.int32)
        keep_len = tf.math.reduce_sum(cumsum_mask)
        traj["actions"] = traj["actions"][:keep_len]
        traj["observations"]["image"] = traj["observations"]["image"][:keep_len]
        traj["observations"]["proprio"] = traj["observations"]["proprio"][:keep_len]
        traj["next_observations"]["image"] = traj["next_observations"]["image"][
            :keep_len
        ]
        traj["next_observations"]["proprio"] = traj["next_observations"]["proprio"][
            :keep_len
        ]
        traj["terminals"] = traj["terminals"][:keep_len]

        return traj

    def _sum_actions_adaptive(self, traj):
        """
        if the gripper action has not changed, we sum over the previus N actions
        until the proprio sum exceeds the threshold
        if the gripper action has changed, we return the current action
        """
        assert self.action_merge_horizon == -1
        threshold = (
            tf.constant(
                # [0.69475555, 0.6719149, 0.6961195, 0.68336874, 0.7066346, 0.50463235],  # mean with standard normal
                # [0.0063365395, 0.008541655, 0.008578023, 0.017812656, 0.020301819, 0.039407436],  # mean
                # [0.0065982887, 0.011018573, 0.009872402, 0.013506467, 0.0193964, 0.044167284], #  mean with merge-2 on 250green
                [
                    0.004217699,
                    0.005272906,
                    0.0060185324,
                    0.009426154,
                    0.0137420595,
                    0.026834767,
                ],  # median with merge-2 on 250 green
                dtype=tf.float32,
            )
            * 2.5
        )
        movement_actions = (
            traj["next_observations"]["proprio"][:, :6]
            - traj["observations"]["proprio"][:, :6]
        )

        if self.relabel_actions:
            # actions are based on proprio

            def scan_fn(acc, i):
                """
                start accumulating action from the first action, and use a sliding
                window to sum until 1) the threshold is reached or 2) the gripper
                action changes. Then we restart the sum from the current action.
                """
                prev_a, kick_out_pointer = acc

                # gripper
                prev_gripper = traj["actions"][i - 1, 6]
                current_gripper = traj["actions"][i, 6]

                # action sums
                prev_sum = tf.ensure_shape(prev_a[:6], (6,))
                current_sum = prev_sum + movement_actions[i, :6]
                current_sum = tf.ensure_shape(current_sum, (6,))

                current_actions = tf.ensure_shape(movement_actions[i, :6], (6,))

                def return_current_action():
                    nonlocal kick_out_pointer
                    a = tf.concat([current_actions, [current_gripper]], axis=0)
                    a = tf.ensure_shape(a, (7,))
                    return a, kick_out_pointer

                def return_summed_action():
                    nonlocal current_sum
                    nonlocal threshold
                    nonlocal kick_out_pointer
                    # check if current_sum exceeds the threshold for half of the action dims
                    while (
                        tf.math.reduce_sum(
                            tf.cast(tf.math.abs(current_sum) >= threshold, tf.float32)
                        )
                        >= 1
                        and kick_out_pointer < i
                    ):
                        # if the sum exceeds the threshold, we kick out the first action
                        # and update the sum
                        current_sum = (
                            current_sum - movement_actions[kick_out_pointer, :6]
                        )
                        current_sum = tf.ensure_shape(current_sum, (6,))
                        kick_out_pointer += 1
                    a = tf.concat([current_sum, [current_gripper]], axis=0)
                    a = tf.ensure_shape(a, (7,))
                    return a, kick_out_pointer

                return tf.cond(
                    tf.not_equal(prev_gripper, current_gripper),
                    return_current_action,
                    return_summed_action,
                )

            new_actions, _ = tf.scan(
                scan_fn,
                tf.range(tf.shape(traj["actions"])[0]),
                (tf.zeros(7), 0),
            )
            traj["actions"] = new_actions

        else:
            # actions are based on recorded actions
            raise NotImplementedError

        return traj

    def _sum_actions(self, traj):
        """
        sum the adjacent self.action_merge_horizon actions together
        this function is used when self.action_merge_horizon > 1
        """
        if self.relabel_actions:
            # actions are based on proprio
            movement_actions = (
                traj["next_observations"]["proprio"][
                    self.action_merge_horizon - 1 :, :6
                ]
                - traj["observations"]["proprio"][
                    : -(self.action_merge_horizon - 1), :6
                ]
            )
            gripper_actions = traj["actions"][self.action_merge_horizon - 1 :, 6]
            traj["actions"] = tf.concat(
                [movement_actions, gripper_actions[:, None]], axis=1
            )

        else:
            # actions are based on recorded actions

            # Workaround using tf.vectorized_map to sum actions and apply conditional logic
            def sum_and_apply_last_gripper(action_slice):
                summed_action = tf.reduce_sum(action_slice, axis=0)
                gripper_val = action_slice[
                    -1, -1
                ]  # the last gripper value in the slice
                return tf.concat([summed_action[:6], [gripper_val]], axis=0)

            # Sum actions over window and apply conditional logic for gripper
            unstacked_summed_actions = tf.vectorized_map(
                sum_and_apply_last_gripper,
                tf.signal.frame(
                    traj["actions"],
                    frame_length=self.action_merge_horizon,
                    frame_step=1,
                    pad_end=False,
                    axis=0,
                ),
            )
            traj["actions"] = unstacked_summed_actions

        # shorten the horizon for everything
        for k in traj["observations"]:
            traj["observations"][k] = traj["observations"][k][
                : -(self.action_merge_horizon - 1)
            ]
        for k in traj["next_observations"]:
            traj["next_observations"][k] = traj["next_observations"][k][
                self.action_merge_horizon - 1 :
            ]
        traj["terminals"] = traj["terminals"][self.action_merge_horizon - 1 :]
        # for k in traj.keys():
        #     if k == "actions":
        #         continue
        #     elif "observation" in k:
        #         for sub_k in traj[k].keys():
        #             traj[k][sub_k] = traj[k][sub_k][self.action_merge_horizon - 1 :]
        #     else:
        #         traj[k] = traj[k][self.action_merge_horizon - 1 :]

        return traj

    def _chunk_act_obs(self, traj):
        traj_len = len(traj["actions"])
        if self.act_pred_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(self.act_pred_horizon), [traj_len, self.act_pred_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.act_pred_horizon]
            )
            # pads by repeating the last action
            chunk_indices = tf.minimum(chunk_indices, traj_len - 1)
            traj["action_chunks"] = tf.gather(traj["actions"], chunk_indices)
        if self.obs_horizon is not None:
            chunk_indices = tf.broadcast_to(
                tf.range(-self.obs_horizon + 1, 1), [traj_len, self.obs_horizon]
            ) + tf.broadcast_to(
                tf.range(traj_len)[:, None], [traj_len, self.obs_horizon]
            )
            # pads by repeating the first observation
            chunk_indices = tf.maximum(chunk_indices, 0)
            traj["obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["observations"]
            )
            traj["next_obs_chunks"] = tf.nest.map_structure(
                lambda x: tf.gather(x, chunk_indices), traj["next_observations"]
            )
        return traj

    def _add_goals(self, traj):
        traj = GOAL_RELABELING_FUNCTIONS[self.goal_relabeling_strategy](
            traj, **self.goal_relabeling_kwargs
        )

        if self.load_language:
            lang_idx = tf.random.uniform(
                shape=[], maxval=len(traj["language"]), dtype=tf.int32
            )
            lang = traj["language"][lang_idx]
            traj["goals"]["language"] = tf.broadcast_to(
                lang, tf.shape(traj["terminals"])
            )
            traj.pop("language")

            # always make the "goal" the last obs so that masking is done
            # properly below
            traj_len = tf.shape(traj["goal_dists"])[0]
            traj["goal_dists"] = traj_len - tf.range(traj_len)

        # after goal relabeling, we can set actions and obs to chunked version
        if "action_chunks" in traj:
            traj["actions"] = traj.pop("action_chunks")
            # set movement actions to 0 after the goal is reached
            new_movement = tf.where(
                (
                    traj["goal_dists"][:, None, None]
                    > tf.range(self.act_pred_horizon)[None, :, None]
                ),  # shape (traj_len, act_pred_horizon, 1)
                traj["actions"][
                    :, :, :-1
                ],  # shape (traj_len, act_pred_horizon, action_dim - 1)
                tf.zeros_like(traj["actions"][0, 0, :-1]),  # shape (action_dim - 1)
            )
            # for gripper actions, repeat the last action after the goal is reached
            new_gripper = tf.where(
                (
                    traj["goal_dists"][:, None]
                    > tf.range(self.act_pred_horizon)[None, :]
                ),  # shape (traj_len, act_pred_horizon)
                traj["actions"][:, :, -1],  # shape (traj_len, act_pred_horizon)
                tf.gather(
                    # shifts `actions` to the right by one, padding with the first action
                    tf.concat(
                        [
                            tf.concat(
                                [
                                    traj["actions"][:1, :1, -1],
                                    traj["actions"][:1, :-1, -1],
                                ],
                                axis=1,
                            ),
                            traj["actions"][:-1, :, -1],
                        ],
                        axis=0,
                    ),
                    # selects the action at index `goal_dists` in the previous action chunk
                    tf.minimum(traj["goal_dists"], self.act_pred_horizon - 1),
                    batch_dims=1,
                )[:, None],
            )
            traj["actions"] = tf.concat([new_movement, new_gripper[:, :, None]], axis=2)
        if "obs_chunks" in traj:
            traj["observations"] = traj.pop("obs_chunks")
            traj["next_observations"] = traj.pop("next_obs_chunks")

        return traj

    def _augment(self, seed, image):
        if self.augment_next_obs_goal_differently:
            sub_seeds = tf.unstack(
                tf.random.stateless_uniform(
                    [3, 2], seed=[seed, seed], minval=None, maxval=None, dtype=tf.int32
                )
            )
        else:
            # use the same seed for obs, next_obs, and goal
            sub_seeds = [[seed, seed]] * 3

        for key, sub_seed in zip(
            ["observations", "next_observations", "goals"], sub_seeds
        ):
            image[key]["image"] = augment(
                image[key]["image"], sub_seed, **self.augment_kwargs
            )
        return image

    def iterator(self):
        return self.tf_dataset.prefetch(tf.data.AUTOTUNE).as_numpy_iterator()
