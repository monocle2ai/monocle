# Capture git metadata of the test environment
import os
from typing import Optional
from git import Repo
from .constants import GIT_COMMIT_HASH_ATTRIBUTE, GIT_RUN_ID_ATTRIBUTE, GIT_WORKFLOW_NAME_ATTRIBUTE

def get_commit_hash() -> str:
    """Get the current git commit hash."""
    if os.getenv("GITHUB_SHA"):
        return os.getenv("GITHUB_SHA")
    repo = Repo(os.getcwd())
    return repo.head.object.hexsha

def get_git_run_id() -> str:
    """Get the current git run ID (GitHub Actions run ID)."""
    return os.getenv("GITHUB_RUN_ID", "local")

def get_git_workflow_name() -> str:
    """Get the current git workflow name (GitHub Actions workflow)."""
    return os.getenv("GITHUB_WORKFLOW", "local")

def get_git_context() -> dict[str, str]:
    """Get a context with git metadata set as attributes."""
    commit_hash = get_commit_hash()
    run_id = get_git_run_id()
    workflow_name = get_git_workflow_name()
    return {
        GIT_COMMIT_HASH_ATTRIBUTE: commit_hash,
        GIT_RUN_ID_ATTRIBUTE: run_id,
        GIT_WORKFLOW_NAME_ATTRIBUTE: workflow_name
    }