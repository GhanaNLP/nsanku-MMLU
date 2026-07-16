#!/usr/bin/env python3
"""
Upload benchmark results to the HF dataset repo so they appear on the leaderboard.

Usage:
    # Upload results from a model evaluation
    python upload_results.py results/openai__gpt-4o/results.json

    # Upload with a custom model name
    python upload_results.py results/openai__gpt-4o/results.json --name "GPT-4o"

Requires: HUGGINGFACE_TOKEN env var with write access to the results repo.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi


RESULTS_REPO = "michsethowusu/ghanaian-mmlu-results"


def main():
    parser = argparse.ArgumentParser(description="Upload results to the leaderboard")
    parser.add_argument("results_file", type=str, help="Path to results.json")
    parser.add_argument("--name", type=str, default=None, help="Custom model name override")
    args = parser.parse_args()

    token = os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: Set HUGGINGFACE_TOKEN or HF_TOKEN env var.", file=sys.stderr)
        sys.exit(1)

    path = Path(args.results_file)
    if not path.exists():
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    if args.name:
        data["model"] = args.name

    model_slug = data["model"].replace("/", "__").replace(" ", "_").lower()
    filename = f"{model_slug}/results.json"

    api = HfApi()
    api.upload_file(
        path_or_fileobj=json.dumps(data, ensure_ascii=False, indent=2).encode(),
        path_in_repo=filename,
        repo_id=RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"Add results for {data.get('model', 'unknown')}",
    )
    print(f"Uploaded to {RESULTS_REPO}/{filename}")
    print(f"View at: https://huggingface.co/spaces/michsethowusu/ghanaian-mmlu-leaderboard")


if __name__ == "__main__":
    main()
