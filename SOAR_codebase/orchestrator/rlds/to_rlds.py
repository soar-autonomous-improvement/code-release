import os
from typing import List
import subprocess

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm


# Function to read frames from a video and store them as a numpy array
def video_to_frames(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    frames = []
    while True:
        # Read next frame
        success, frame = cap.read()
        if not success:
            break

        # Convert BGR to RGB as OpenCV uses BGR by default
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Append the RGB frame to the frames list
        frames.append(frame_rgb)

    # Close the video file
    cap.release()

    # Convert the list of frames to a numpy array
    frames_array = np.stack(frames, axis=0)

    return frames_array


def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )


def port_trajectories_to_rlds(config, trajs_to_port: List[int] = None):
    # Delete the folder we're writing to, if it exists (so we don't re-upload things unecessarily)
    subprocess.run(
        [
            "rm",
            "-rf",
            config["rlds_params"]["rlds_dataset_path"]
        ]
    )

    # Re-make the directory
    os.makedirs(config["rlds_params"]["rlds_dataset_path"], exist_ok=True)

    if trajs_to_port is None:
        trajs_to_port = [int(name[4:]) for name in os.listdir(config["general_params"]["video_save_path"]) if os.path.isdir(os.path.join(config["general_params"]["video_save_path"], name))]

    pbar = tqdm(total=len(trajs_to_port))
    for traj_idx in trajs_to_port:
        src_folder = os.path.join(config["general_params"]["video_save_path"], "traj" + str(traj_idx))

        try:
            # Actions, observations, and goals
            traj_rel_actions = np.load(os.path.join(src_folder, "actions.npy")).astype(np.float32)
            traj_observations = video_to_frames(os.path.join(src_folder, "trajectory.mp4")).astype(np.uint8)
            traj_proprio = np.load(os.path.join(src_folder, "eef_poses.npy")).astype(np.float32)
            traj_commanded_goals = video_to_frames(os.path.join(src_folder, "goals.mp4")).astype(np.uint8)

            with open(os.path.join(src_folder, "language_task.txt")) as f:
                prompt = f.readlines()[0].strip().lower()
            traj_language = [prompt for _ in range(len(traj_rel_actions))]
            
            with open(os.path.join(src_folder, "success.txt")) as f:
                success = f.readlines()[0].strip().lower()
                assert success in ("true", "false")
                success = success == "true"
            
            with open(os.path.join(src_folder, "time.txt")) as f:
                time = f.readlines()[0].strip()
            
            with open(os.path.join(src_folder, "robot_id.txt")) as f:
                robot_id = f.readlines()[0].strip()

            output_tfrecord_path = os.path.join(
                config["rlds_params"]["rlds_dataset_path"],
                "traj" + str(traj_idx) + ".tfrecord",
            )
            assert traj_commanded_goals.shape == traj_observations.shape, (traj_commanded_goals.shape, traj_observations.shape)
            with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "observations/images0": tensor_feature(
                                traj_observations[:-1].astype(np.uint8)
                            ),
                            "observations/state": tensor_feature(
                                traj_proprio[:-1].astype(np.float32)
                            ),
                            "next_observations/images0": tensor_feature(
                                traj_observations[1:].astype(np.uint8)
                            ),
                            "next_observations/state": tensor_feature(
                                traj_proprio[1:].astype(np.float32)
                            ),
                            "language": tensor_feature(traj_language),
                            "actions": tensor_feature(
                                traj_rel_actions[:-1].astype(np.float32)
                            ),
                            "commanded_goals": tensor_feature(
                                traj_commanded_goals[:-1].astype(np.uint8)
                            ),
                            "terminals": tensor_feature(
                                np.zeros(len(traj_rel_actions)-1, dtype=bool)
                            ),
                            "truncates": tensor_feature(
                                np.zeros(len(traj_rel_actions)-1, dtype=bool)
                            ),
                            "vlm_success": tensor_feature(
                                np.array([success], dtype=bool)
                            ),
                            "time": tensor_feature(
                                np.array([time], dtype=str)
                            ),
                            "robot_id": tensor_feature(
                                np.array([robot_id], dtype=str)
                            ),
                        }
                    )
                )
                writer.write(example.SerializeToString())
        except KeyboardInterrupt:
            print("keyboard interrupt")
            exit()
        except:
            print("Input/output error (probably), skipping...")
        
        pbar.update(1)
    pbar.close()

