import os
import time
import argparse
from huggingface_hub import upload_file, create_repo

def upload_folder(local_folder, repo_id, repo_type="model"):
    for filename in os.listdir(local_folder):
        filepath = os.path.join(local_folder, filename)
        if os.path.isfile(filepath):
            success = False
            attempts = 0
            while not success and attempts < 3:
                try:
                    print(f"Uploading {filename} (attempt {attempts + 1})...")
                    upload_file(
                        path_or_fileobj=filepath,
                        path_in_repo=filename,
                        repo_id=repo_id,
                        repo_type=repo_type,
                    )
                    success = True
                except Exception as e:
                    print(f"⚠️ Error uploading {filename} Error: {e}. Retrying...")
                    attempts += 1
                    time.sleep(5)
            if not success:
                print(f"❌ Failed to upload {filename} after 3 attempts.")
    print("✅ Upload process finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name (e.g., username/model-name)")
    parser.add_argument("--checkpoint_folder", type=str, default="/leonardo_work/EUHPC_E03_068/DCFT_shared/checkpoints", help="Path to the checkpoint folder to upload")
    args = parser.parse_args()

    if "mlfoundations-dev/" not in args.model_name:
        # Support legacy DCAgent/ prefix by normalizing to bare name
        args.model_name = args.model_name.replace("DCAgent/", "")
    local_folder = f"{args.checkpoint_folder}/{args.model_name}"

    try:
        create_repo(
            repo_id=f"mlfoundations-dev/{args.model_name}",
            repo_type="model",
            exist_ok=True,
        )
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        exit(1)

    upload_folder(local_folder=local_folder, repo_id=f"mlfoundations-dev/{args.model_name}")
