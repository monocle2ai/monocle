# Capture git metadata of the test environment
from datetime import datetime, timezone
import os
from git import Repo
from .constants import GIT_COMMIT_HASH_ATTRIBUTE, GIT_RUN_ID_ATTRIBUTE, GIT_WORKFLOW_NAME_ATTRIBUTE

def get_commit_hash() -> str:
    """Get the current git commit hash."""
    if os.getenv("GITHUB_SHA"):
        return os.getenv("GITHUB_SHA")
    try:
        repo = Repo(os.getcwd(), search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return "unknown"

def get_git_run_id() -> str:
    """Get the current git run ID (GitHub Actions run ID) or current timestamp for local runs."""
    return os.getenv("GITHUB_RUN_ID", datetime.now().isoformat())

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

def get_repo_name() -> str:
    """Get the name of the git org + repository"""
    try:
        repo = Repo(os.getcwd(), search_parent_directories=True)
        remote_url = repo.remotes.origin.url

        if remote_url.startswith('git@'):
            # SSH: git@github.com:org/repo.git
            path = remote_url.split(':')[1].replace('/', '-')
        else:
            # HTTPS: https://github.com/org/repo.git
            path = '-'.join(remote_url.split('/')[-2:])

        # Remove .git extension if present
        repo_name = os.path.splitext(path)[0]
        return repo_name  # Returns "org-repo"
    except Exception:
        return None