from typing import List
from huggingface_hub import HfApi


class DCLMShardDownloader:
    """Prepare download URL list for DCLM baseline on ZIH."""

    def __init__(self, repo_id: str = "mlfoundations/dclm-baseline-1.0"):
        self.repo_id = repo_id
        self.api = HfApi()

    def list_all_shards(self) -> List[str]:
        """List all .jsonl.zst shard files in the dataset."""
        print(f"Scanning {self.repo_id} for shard files...", flush=True)

        try:
            repo_files = self.api.list_repo_files(
                repo_id=self.repo_id,
                repo_type="dataset",
            )
        except Exception as e:
            print(f"Error accessing repository: {e}", flush=True)
            raise

        shard_paths = [f for f in repo_files if f.endswith(".jsonl.zst")]
        print(f"Found {len(shard_paths)} shard files", flush=True)
        return shard_paths

    def write_url_list(self, out_path: str = "urls.txt"):
        """Write HTTPS download URLs for all shards to a text file."""
        shard_paths = self.list_all_shards()
        base = f"https://huggingface.co/datasets/{self.repo_id}/resolve/main"

        print(f"Writing URLs to {out_path} ...", flush=True)
        with open(out_path, "w") as f:
            for rel in shard_paths:
                f.write(f"{base}/{rel}\n")

        print(f"Done. Wrote {len(shard_paths)} URLs.", flush=True)


if __name__ == "__main__":
    dl = DCLMShardDownloader()
    dl.write_url_list("urls.txt")