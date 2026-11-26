"""Project configuration, shared .env resolution across Worktrees, and device detection."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml
from dotenv import load_dotenv

CANONICAL_COLUMNS: list[str] = [
    "id",
    "source",
    "category",
    "title",
    "meta_tags",
    "content",
    "posted_at",
    "deadline",
    "url",
]

FEEDBACK_COLUMNS: list[str] = [
    "notice_id",
    "label",
    "source",
    "timestamp",
    "comment",
]


def find_repo_root(start: str | Path | None = None) -> Path:
    """Return the root directory of the current repository or worktree."""
    current = Path(start or Path(__file__).resolve().parent.parent).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return current


def find_shared_env_path(start: str | Path | None = None) -> Path | None:
    """Locate .env in the current repo or fall back to the main worktree's shared .env."""
    repo_root = find_repo_root(start)
    local_env = repo_root / ".env"
    if local_env.is_file():
        return local_env

    try:
        common_dir = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if common_dir:
            common_path = Path(common_dir)
            if not common_path.is_absolute():
                common_path = (repo_root / common_path).resolve()
            shared_env = common_path.parent / ".env"
            if shared_env.is_file():
                return shared_env
    except (OSError, subprocess.SubprocessError):
        pass

    return None


def load_env(start: str | Path | None = None, override: bool = False) -> Path | None:
    """Load environment variables from the local or shared .env file."""
    env_path = find_shared_env_path(start)
    if env_path is not None:
        load_dotenv(dotenv_path=env_path, override=override)
    return env_path


def load_profile(path: str | Path | None = None) -> dict[str, Any]:
    """Load and return the structured user profile and rubric from YAML."""
    profile_path = Path(path) if path is not None else find_repo_root() / "config" / "profile.yaml"
    with profile_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid profile YAML at {profile_path}")
    return data


def get_device(preferred: str | None = None) -> str:
    """Return 'cuda' when available or fall back to 'cpu', honoring explicit preference."""
    if preferred is not None:
        normalized = preferred.strip().lower()
        if normalized == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if normalized in {"cuda", "cpu"}:
            return normalized
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_env_flag(name: str) -> bool:
    """Check whether an environment variable is set and non-empty without exposing its value."""
    return bool(os.getenv(name, "").strip())
