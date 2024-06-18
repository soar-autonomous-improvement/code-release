import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import re
import subprocess
import time

from absl import app, flags
from tqdm import tqdm
import yaml
from yamlinclude import YamlIncludeConstructor

from orchestrator.rlds import to_rlds

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config_dir",
    None,
    "Path to config.yaml",
    required=True,
)


def extract_and_identify_n(file_path):
    """
    rlds uploads are in the format */traj{n}.tfrecord
    this function returns n given a file path
    """
    # Define the regex pattern to match the required format
    pattern = r".*/+traj(\d+)\.tfrecord"
    
    # Search for the pattern in the given file path
    match = re.search(pattern, file_path)
    
    # If a match is found, extract 'n'
    if match:
        return int(match.group(1))  # Return 'n' as an integer
    else:
        return None  # No match found, return None


def main(_):
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=FLAGS.config_dir)
    with open(os.path.join(FLAGS.config_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Every config["rlds_params"]["cloud_upload_wait_time"] minutes we will write 
    # autonomously collected trajectories into RLDS format and upload to the google 
    # cloud bucket
    while True:
        # We will first determine which trajectories we need to convert to RLDS and then upload
        # We can do this my getting a list of trajectories from the cloud bucket, and see which 
        # trajectories have already been uploaded
        process = subprocess.Popen(["gsutil", "ls", config["rlds_params"]["cloud_upload_dir"]], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
        out, err = process.communicate()
        return_output = out.decode("ascii")
        trajectories_uploaded = return_output.split("\n")
        traj_idxs_uploaded = []
        for t in trajectories_uploaded:
            if ".tfrecord" in t:
                traj_id = extract_and_identify_n(t)
                assert traj_id is not None, "Trajectory ID not found in the file path"
                traj_idxs_uploaded.append(traj_id)
        traj_idxs_uploaded = set(traj_idxs_uploaded)
        
        all_trajs = os.listdir(config["general_params"]["video_save_path"])
        trajs_to_port = []
        for traj in all_trajs:
            # confirm it is a traj
            if "traj" in traj:
                traj_idx = int(traj[4:])
                if traj_idx not in traj_idxs_uploaded:
                    trajs_to_port.append(traj_idx)

        to_rlds.port_trajectories_to_rlds(config, trajs_to_port)
        print("Uploading local dataset to google bucket...")
        subprocess.run(
            [
                "gsutil",
                "-m",
                "cp",
                os.path.join(config["rlds_params"]["rlds_dataset_path"], "*.tfrecord"),
                config["rlds_params"]["cloud_upload_dir"],
            ]
        )
        print("Waiting", config["rlds_params"]["cloud_upload_wait_time"], "minutes...")
        for _ in tqdm(range(60 * config["rlds_params"]["cloud_upload_wait_time"])):
            time.sleep(1)


if __name__ == "__main__":
    app.run(main)
