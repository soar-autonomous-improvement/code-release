import yaml
from yamlinclude import YamlIncludeConstructor
from absl import app, flags
import subprocess
import time
import os

from absl import app, flags
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "config_dir",
    None,
    "Path to config directory",
    required=True,
)


def run_with_retry(command, retry_count=3, search_term="NotFoundException", cleanup_command=None):
    """
    Runs a subprocess command with retries if the specified search_term is found 
    in the command's output.
    We use this to retry the gsutil download command if it fails due to a NotFoundException,
    which can happen because we are downloading while the training run is simultaneously
    writing to the same location.

    Args:
        command (list): The command and its arguments to be executed.
        retry_count (int): Number of times to retry the command.
        search_term (str): The term to search for in the command's output.
        cleanup_command (list): The command to run to "clean up" if the search term is found.

    Returns:
        subprocess.CompletedProcess: The result of the subprocess command execution.
    """
    for attempt in range(retry_count):
        # Run the command with output capture
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Check if the search term is in the output or error
        if search_term in result.stdout or search_term in result.stderr:
            print(f"\033[91m'{search_term}' found in attempt {attempt + 1}, retrying...\033[0m")
            if cleanup_command is not None:
                subprocess.run(cleanup_command)
            continue  # Retry the command
        
        # If the term was not found or command succeeded, break out of the loop
        break
    
    return result


def main(_):
    """
    Every pre-defined period of time, pull the latest model checkpoint from google cloud
    """
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=FLAGS.config_dir)
    with open(os.path.join(FLAGS.config_dir, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Make sure destination directory exists
    os.makedirs(config["checkpoint_sync_params"]["dest_checkpoint_path"], exist_ok=True)

    def wait_till_next_iteration():
        print("Waiting", config["checkpoint_sync_params"]["wait_time"], "minutes...")
        for _ in tqdm(range(60 * config["checkpoint_sync_params"]["wait_time"])):
            time.sleep(1)

    while True:
        # Download the latest checkpoint
        print("Downloading latest finetuned checkpoint...")
        command = [
            # "timeout",
            # "2m", # fixes a bug where download takes forever
            "gsutil",
            "cp",
            "-r",
            config["checkpoint_sync_params"]["src_checkpoint_root_folder"],
            config["checkpoint_sync_params"]["dest_checkpoint_path"]
        ]
        cleanup_command = [
            "rm",
            "-r",
            os.path.join(
                config["checkpoint_sync_params"]["dest_checkpoint_path"],
                config["checkpoint_sync_params"]["src_checkpoint_root_folder"].split("/")[-1]  # by convention this is checkpoint_0
            )
        ]
        result = run_with_retry(command, cleanup_command=cleanup_command)
        print("Command output:", result.stdout)
        print("Command error (if any):", result.stderr)
        
        wait_till_next_iteration()


if __name__ == "__main__":
    app.run(main)
