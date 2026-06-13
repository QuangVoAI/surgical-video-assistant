from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.jsonl",
    "*.jpg",
    "*.jpeg",
    "*.png",
    "*.webp",
    "*.parquet",
    "*.txt",
    "*.md",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face dataset snapshot for raw parsing.")
    parser.add_argument("--repo-id", default="introvoyz041/SurgMLLMBench")
    parser.add_argument("--local-dir", type=Path, default=Path("data/raw/SurgMLLMBench"))
    parser.add_argument("--revision", default=None)
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Download every file instead of the common surgical image/annotation extensions.",
    )
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    downloaded_path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
        token=token,
        allow_patterns=None if args.all_files else DEFAULT_ALLOW_PATTERNS,
    )
    print(f"Downloaded {args.repo_id} to {downloaded_path}")
    print("Next step:")
    print(f"python -m src.data.prepare --dataset surgmllmbench --raw {args.local_dir} --out data/processed")


if __name__ == "__main__":
    main()
