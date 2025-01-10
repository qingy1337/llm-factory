import os
import re
import time
import shutil
from huggingface_hub import login, create_repo, upload_folder

# Regex pattern to match checkpoint folders
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")

# Set to track uploaded checkpoints
uploaded_checkpoints = set()

def find_new_checkpoint():
    """
    Finds the checkpoint folder with the highest number that has not been uploaded yet.
    """
    items = os.listdir(".")
    checkpoints = [
        (int(match.group(1)), item)
        for item in items
        if (match := CHECKPOINT_PATTERN.match(item)) and item not in uploaded_checkpoints
    ]
    if checkpoints:
        checkpoints.sort(reverse=True)
        return checkpoints[0]  # Return the highest checkpoint
    return None

def upload_checkpoint(folder_name, checkpoint_number):
    """
    Uploads the checkpoint folder to Hugging Face and deletes the folder locally.
    """
    repo_name = f"qingy2024/Qwarkstar-4B-Instruct"
    print(f"Uploading {folder_name} to Hugging Face as {repo_name}...")

    # Create the Hugging Face repository
    create_repo(repo_name, exist_ok=True)

    # Upload the folder to the repository
    upload_folder(
        folder_path=folder_name,
        repo_id=repo_name,
        commit_message=f"Upload checkpoint {checkpoint_number}"
    )

    # Delete the local folder
    print(f"Deleting local folder: {folder_name}")
    shutil.rmtree(folder_name)
    print(f"Checkpoint {folder_name} successfully uploaded and deleted.")

def main():
    """
    Main loop to monitor and process new checkpoints.
    """
    print("Starting checkpoint monitor...")
    while True:
        new_checkpoint = find_new_checkpoint()

        if new_checkpoint:
            time.sleep(30)
            checkpoint_number, folder_name = new_checkpoint
            try:
                upload_checkpoint(folder_name, checkpoint_number)
                uploaded_checkpoints.add(folder_name)
            except Exception as e:
                print(f"Error uploading {folder_name}: {e}")
        else:
            print("No new checkpoints found. Checking again in 5 minutes...")

        # Wait for 2 minutes before checking again
        time.sleep(120)

if __name__ == "__main__":
    main()
