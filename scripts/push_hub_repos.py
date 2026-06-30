#!/usr/bin/env python3
"""Upload 4 separate Hugging Face repos and remove old combined repos."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "reports" / "models"
STAGING = ROOT / "hub_upload"

OLD_DATASET_REPO = "ganga4364/tibetan-outline-boundary-snippets"
OLD_MODEL_REPO = "ganga4364/tibetan-outline-boundary-crf"

DATASET_REPOS = {
    "ganga4364/tibetan-outline-boundary-snippets-full": {
        "json_path": DATA / "breakpoints_context_snippets.json",
        "json_name": "breakpoints_context_snippets.json",
        "staging": STAGING / "snippets-full",
    },
    "ganga4364/tibetan-outline-boundary-snippets-unique": {
        "json_path": DATA / "breakpoints_context_snippets_unique.json",
        "json_name": "breakpoints_context_snippets_unique.json",
        "staging": STAGING / "snippets-unique",
    },
}

MODEL_REPOS = {
    "ganga4364/tibetan-outline-boundary-crf-full": {
        "source": MODELS / "boundary_crf.pkl",
        "staging": STAGING / "crf-full",
    },
    "ganga4364/tibetan-outline-boundary-crf-unbiased": {
        "source": MODELS / "boundary_crf_holdout.pkl",
        "staging": STAGING / "crf-unbiased",
    },
}


def require_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("Set HF_TOKEN before running this script.")
    return token


def load_snippets(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"{path} must be a JSON array of strings")
    return [str(item) for item in data]


def run_hf(args: list[str], token: str) -> None:
    cmd = ["hf", *args, "--token", token]
    print("+", " ".join(cmd[:-2] + ["--token", "***"]))
    subprocess.run(cmd, check=True)


def push_dataset_repo(repo_id: str, cfg: dict, token: str) -> None:
    from datasets import Dataset
    from huggingface_hub import HfApi

    snippets = load_snippets(cfg["json_path"])
    staging = cfg["staging"]
    json_name = cfg["json_name"]

    print(f"\n=== Dataset: {repo_id} ({len(snippets):,} snippets) ===")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    ds = Dataset.from_dict({"snippet": snippets})
    ds.push_to_hub(
        repo_id,
        split="train",
        token=token,
        commit_message=f"Add train split ({len(snippets):,} snippets) and JSON mirror",
    )

    readme = staging / "README.md"
    json_mirror = staging / json_name
    if not readme.exists():
        raise FileNotFoundError(readme)
    if not json_mirror.exists():
        shutil.copy2(cfg["json_path"], json_mirror)

    run_hf(
        [
            "upload",
            repo_id,
            str(readme),
            "README.md",
            "--repo-type",
            "dataset",
            "--commit-message",
            "Update dataset card",
        ],
        token,
    )
    run_hf(
        [
            "upload",
            repo_id,
            str(json_mirror),
            json_name,
            "--repo-type",
            "dataset",
            "--commit-message",
            f"Add {json_name} mirror",
        ],
        token,
    )


def push_model_repo(repo_id: str, cfg: dict, token: str) -> None:
    from huggingface_hub import HfApi

    staging = cfg["staging"]
    source = cfg["source"]
    target_pkl = staging / "boundary_crf.pkl"
    readme = staging / "README.md"

    print(f"\n=== Model: {repo_id} ===")

    if not source.exists():
        raise FileNotFoundError(source)
    if not readme.exists():
        raise FileNotFoundError(readme)

    staging.mkdir(parents=True, exist_ok=True)
    if not target_pkl.exists() or target_pkl.stat().st_size != source.stat().st_size:
        shutil.copy2(source, target_pkl)

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    run_hf(
        [
            "upload",
            repo_id,
            str(staging),
            ".",
            "--repo-type",
            "model",
            "--include",
            "README.md",
            "boundary_crf.pkl",
            "--commit-message",
            "Add CRF model and model card",
        ],
        token,
    )


def delete_old_repos(token: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    for repo_id, repo_type in (
        (OLD_DATASET_REPO, "dataset"),
        (OLD_MODEL_REPO, "model"),
    ):
        print(f"\n=== Delete old repo: {repo_id} ({repo_type}) ===")
        try:
            api.delete_repo(repo_id, repo_type=repo_type, token=token)
            print(f"Deleted {repo_id}")
        except Exception as exc:
            print(f"Could not delete {repo_id}: {exc}", file=sys.stderr)


def main() -> None:
    token = require_token()

    for repo_id, cfg in DATASET_REPOS.items():
        push_dataset_repo(repo_id, cfg, token)

    for repo_id, cfg in MODEL_REPOS.items():
        push_model_repo(repo_id, cfg, token)

    delete_old_repos(token)
    print("\nAll uploads complete.")


if __name__ == "__main__":
    main()
