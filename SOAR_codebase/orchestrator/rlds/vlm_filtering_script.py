import os
import shutil
import sys
from tqdm import tqdm


def main():
    try:
        source_dir = sys.argv[1]
    except IndexError:
        print("Usage: python script.py <source_dir>")
        sys.exit(1)
    
    success_dir = source_dir + "_success"
    failure_dir = source_dir + "_failure"

    # recreate these folders
    os.makedirs(success_dir, exist_ok=False)
    os.makedirs(failure_dir, exist_ok=False)
    
    subdirectories = [
        os.path.join(source_dir, d) for d in os.listdir(source_dir) 
        if os.path.isdir(os.path.join(source_dir, d))
    ]
    success_count = 0
    failure_count = 0

    for subdirectory in tqdm(subdirectories):
        # recorded VLM success
        with open(os.path.join(subdirectory, "success.txt")) as f:
            success = f.readlines()[0].strip().lower()
            assert success in ("true", "false")
            success = success == "true"
        
        if success:
            shutil.copytree(subdirectory, os.path.join(success_dir, os.path.basename(subdirectory)))
            #os.symlink(
            #    os.path.abspath(subdirectory),
            #    os.path.abspath(os.path.join(success_dir, os.path.basename(subdirectory)))
            #)
            success_count += 1
        else:
            shutil.copytree(subdirectory, os.path.join(failure_dir, os.path.basename(subdirectory)))
            #os.symlink(
            #    os.path.abspath(subdirectory),
            #    os.path.abspath(os.path.join(failure_dir, os.path.basename(subdirectory)))
            #)
            failure_count += 1

    print(f"Subdirectories with success: {success_count}")
    print(f"Subdirectories with failure: {failure_count}")


if __name__ == "__main__":
    main()
