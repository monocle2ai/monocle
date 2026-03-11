# Capture git metadata of the test environment
import os
from datetime import datetime, timezone

from git import Repo

from .constants import (
    GIT_COMMIT_HASH_ATTRIBUTE,
    GIT_RUN_ID_ATTRIBUTE,
    GIT_WORKFLOW_NAME_ATTRIBUTE,
    GITHUB_RUN_ID,
    GITHUB_SHA,
    GITHUB_WORKFLOW,
    JENKINS_BUILD_ID,
    JENKINS_BUILD_NUMBER,
    JENKINS_BUILD_URL,
    JENKINS_BUILD_URL_ATTRIBUTE,
    JENKINS_GIT_COMMIT,
    JENKINS_JOB_NAME,
    JENKINS_NODE_NAME,
    JENKINS_NODE_NAME_ATTRIBUTE,
    JENKINS_URL,
    JENKINS_URL_ATTRIBUTE,
    LOCAL_RUN_ID,
)


def get_commit_hash() -> str:
    """Get the current git commit hash."""
    # Check Jenkins Git plugin environment variable first
    jenkins_git_commit = os.getenv(JENKINS_GIT_COMMIT)
    if jenkins_git_commit:
        return jenkins_git_commit

    # Check GitHub Actions
    if os.getenv(GITHUB_SHA):
        return os.getenv(GITHUB_SHA)

    # Fallback to local git repo
    try:
        repo = Repo(os.getcwd(), search_parent_directories=True)
        return repo.head.object.hexsha
    except Exception:
        return "unknown"


def get_git_run_id() -> str:
    """Get the current git run ID (GitHub Actions run ID) or current timestamp for local runs."""
    # Check for Jenkins build identifiers
    jenkins_build_number = os.getenv(JENKINS_BUILD_NUMBER)
    jenkins_build_id = os.getenv(JENKINS_BUILD_ID)
    if jenkins_build_number and jenkins_build_id:
        return f"jenkins_{jenkins_build_number}_{jenkins_build_id}"

    # Check for GitHub Actions
    github_run_id = os.getenv(GITHUB_RUN_ID)
    if github_run_id:
        return f"github_{github_run_id}"

    # Check for cached local run ID
    local_run_id = os.getenv(LOCAL_RUN_ID)
    if local_run_id:
        return local_run_id

    # Fallback: set LOCAL_RUN_ID for subsequent calls in the same process
    run_id = datetime.now(timezone.utc).isoformat()
    os.environ[LOCAL_RUN_ID] = run_id
    return run_id


def get_git_workflow_name() -> str:
    """Get the current git workflow name (GitHub Actions workflow, Jenkins job, or local)."""
    # Check for Jenkins job name
    jenkins_job_name = os.getenv(JENKINS_JOB_NAME)
    if jenkins_job_name:
        return jenkins_job_name

    # Check for GitHub Actions workflow
    github_workflow = os.getenv(GITHUB_WORKFLOW)
    if github_workflow:
        return github_workflow

    # Default to local
    return "local"


def get_git_context() -> dict[str, str]:
    """Get a context with git metadata set as attributes."""
    commit_hash = get_commit_hash()
    run_id = get_git_run_id()
    workflow_name = get_git_workflow_name()

    context = {
        GIT_COMMIT_HASH_ATTRIBUTE: commit_hash,
        GIT_RUN_ID_ATTRIBUTE: run_id,
        GIT_WORKFLOW_NAME_ATTRIBUTE: workflow_name,
    }

    # Add Jenkins-specific metadata if running in Jenkins
    jenkins_url = os.getenv(JENKINS_URL)
    build_url = os.getenv(JENKINS_BUILD_URL)
    node_name = os.getenv(JENKINS_NODE_NAME)

    if jenkins_url:
        context[JENKINS_URL_ATTRIBUTE] = jenkins_url
    if build_url:
        context[JENKINS_BUILD_URL_ATTRIBUTE] = build_url
    if node_name:
        context[JENKINS_NODE_NAME_ATTRIBUTE] = node_name

    return context


def get_repo_name() -> str:
    """Get the name of the git org + repository"""
    try:
        repo = Repo(os.getcwd(), search_parent_directories=True)
        remote_url = repo.remotes.origin.url

        if remote_url.startswith("git@"):
            # SSH: git@github.com:org/repo.git
            path = remote_url.split(":")[1].replace("/", "-")
        else:
            # HTTPS: https://github.com/org/repo.git
            path = "-".join(remote_url.split("/")[-2:])

        # Remove .git extension if present
        repo_name = os.path.splitext(path)[0]
        return repo_name  # Returns "org-repo"
    except Exception:
        return None
