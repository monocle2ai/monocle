"""Unit tests for gitutils module - CI/CD environment detection"""

import os

import pytest
from monocle_test_tools.constants import (
    GIT_BRANCH_ATTRIBUTE,
    GIT_COMMIT_HASH_ATTRIBUTE,
    GIT_REPO_ATTRIBUTE,
    GIT_RUN_ID_ATTRIBUTE,
    GIT_WORKFLOW_NAME_ATTRIBUTE,
    JENKINS_BUILD_URL_ATTRIBUTE,
    JENKINS_NODE_NAME_ATTRIBUTE,
    JENKINS_URL_ATTRIBUTE,
)
from monocle_test_tools.gitutils import (
    get_commit_hash,
    get_git_branch,
    get_git_context,
    get_git_run_id,
    get_git_workflow_name,
)


class TestGitUtilsJenkinsDetection:
    """Test Jenkins CI/CD environment detection"""

    @pytest.fixture(autouse=True)
    def cleanup_env_vars(self):
        """Clean up environment variables before and after each test"""
        # Store original values
        original_env = {}
        jenkins_vars = [
            "BUILD_NUMBER",
            "BUILD_ID",
            "JOB_NAME",
            "BUILD_URL",
            "JENKINS_URL",
            "GIT_COMMIT",
            "NODE_NAME",
            "GIT_BRANCH",
        ]
        github_vars = [
            "GITHUB_WORKFLOW",
            "GITHUB_RUN_ID",
            "GITHUB_SHA",
        ]

        for var in jenkins_vars + github_vars + ["LOCAL_RUN_ID"]:
            original_env[var] = os.getenv(var)
            if var in os.environ:
                del os.environ[var]

        yield

        # Restore original values
        for var, value in original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_jenkins_workflow_name_detection(self):
        """Test that Jenkins JOB_NAME is detected as workflow name"""
        os.environ["JOB_NAME"] = "my-jenkins-job"

        workflow_name = get_git_workflow_name()
        assert workflow_name == "my-jenkins-job"

    def test_jenkins_run_id_detection(self):
        """Test that Jenkins BUILD_NUMBER and BUILD_ID create unique run ID"""
        os.environ["BUILD_NUMBER"] = "42"
        os.environ["BUILD_ID"] = "2026-03-08_10-23-45"

        run_id = get_git_run_id()
        assert run_id == "jenkins_42_2026-03-08_10-23-45"

    def test_jenkins_partial_build_info(self):
        """Test that Jenkins run ID requires both BUILD_NUMBER and BUILD_ID"""
        # Only BUILD_NUMBER, no BUILD_ID
        os.environ["BUILD_NUMBER"] = "42"

        run_id = get_git_run_id()
        # Should fall back to local run ID (timestamp format)
        assert not run_id.startswith("jenkins_")

    def test_jenkins_commit_hash_detection(self):
        """Test that Jenkins GIT_COMMIT is used for commit hash"""
        os.environ["GIT_COMMIT"] = "abc123def456"

        commit_hash = get_commit_hash()
        assert commit_hash == "abc123def456"

    def test_jenkins_full_context(self):
        """Test complete Jenkins context with all metadata"""
        os.environ["JOB_NAME"] = "travel-agent-tests"
        os.environ["BUILD_NUMBER"] = "100"
        os.environ["BUILD_ID"] = "2026-03-08_14-30-00"
        os.environ["GIT_COMMIT"] = "fedcba987654"
        os.environ["JENKINS_URL"] = "https://jenkins.example.com/"
        os.environ["BUILD_URL"] = (
            "https://jenkins.example.com/job/travel-agent-tests/100/"
        )
        os.environ["NODE_NAME"] = "jenkins-worker-1"

        context = get_git_context()

        assert context[GIT_WORKFLOW_NAME_ATTRIBUTE] == "travel-agent-tests"
        assert context[GIT_RUN_ID_ATTRIBUTE] == "jenkins_100_2026-03-08_14-30-00"
        assert context[GIT_COMMIT_HASH_ATTRIBUTE] == "fedcba987654"
        assert context[JENKINS_URL_ATTRIBUTE] == "https://jenkins.example.com/"
        assert (
            context[JENKINS_BUILD_URL_ATTRIBUTE]
            == "https://jenkins.example.com/job/travel-agent-tests/100/"
        )
        assert context[JENKINS_NODE_NAME_ATTRIBUTE] == "jenkins-worker-1"

    def test_jenkins_minimal_context(self):
        """Test Jenkins context with minimal required variables"""
        os.environ["JOB_NAME"] = "minimal-job"
        os.environ["BUILD_NUMBER"] = "1"
        os.environ["BUILD_ID"] = "2026-03-08_09-00-00"

        context = get_git_context()

        # Should have basic git attributes
        assert context[GIT_WORKFLOW_NAME_ATTRIBUTE] == "minimal-job"
        assert context[GIT_RUN_ID_ATTRIBUTE] == "jenkins_1_2026-03-08_09-00-00"

        # Should NOT have Jenkins-specific attributes if not set
        assert JENKINS_URL_ATTRIBUTE not in context
        assert JENKINS_BUILD_URL_ATTRIBUTE not in context
        assert JENKINS_NODE_NAME_ATTRIBUTE not in context


class TestGitUtilsLocalDetection:
    """Test local environment detection (developer laptop)"""

    @pytest.fixture(autouse=True)
    def cleanup_env_vars(self):
        """Clean up environment variables before and after each test"""
        original_env = {}
        all_vars = [
            "BUILD_NUMBER",
            "BUILD_ID",
            "JOB_NAME",
            "BUILD_URL",
            "JENKINS_URL",
            "GIT_COMMIT",
            "NODE_NAME",
            "GITHUB_WORKFLOW",
            "GITHUB_RUN_ID",
            "GITHUB_SHA",
            "LOCAL_RUN_ID",
        ]

        for var in all_vars:
            original_env[var] = os.getenv(var)
            if var in os.environ:
                del os.environ[var]

        yield

        for var, value in original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_local_workflow_name_fallback(self):
        """Test that 'local' is returned when no CI environment is detected"""
        workflow_name = get_git_workflow_name()
        assert workflow_name == "local"

    def test_local_run_id_generation(self):
        """Test that local run ID is generated with timestamp"""
        run_id = get_git_run_id()

        # Should be ISO timestamp format
        assert ":" in run_id  # Timestamps have colons
        assert "T" in run_id  # ISO format has T separator
        assert not run_id.startswith("jenkins_")
        assert not run_id.startswith("github_")

    def test_local_run_id_caching(self):
        """Test that local run ID is cached in environment"""
        run_id_1 = get_git_run_id()
        run_id_2 = get_git_run_id()

        # Should return same ID (cached)
        assert run_id_1 == run_id_2
        assert os.getenv("LOCAL_RUN_ID") == run_id_1

    def test_local_context(self):
        """Test local context has basic git attributes only"""
        context = get_git_context()

        assert context[GIT_WORKFLOW_NAME_ATTRIBUTE] == "local"
        assert GIT_RUN_ID_ATTRIBUTE in context
        assert GIT_COMMIT_HASH_ATTRIBUTE in context

        # Should NOT have CI-specific attributes
        assert JENKINS_URL_ATTRIBUTE not in context
        assert JENKINS_BUILD_URL_ATTRIBUTE not in context


class TestGitUtilsPriorityOrder:
    """Test priority order: Jenkins > GitHub Actions > Local"""

    @pytest.fixture(autouse=True)
    def cleanup_env_vars(self):
        """Clean up environment variables before and after each test"""
        original_env = {}
        all_vars = [
            "BUILD_NUMBER",
            "BUILD_ID",
            "JOB_NAME",
            "BUILD_URL",
            "JENKINS_URL",
            "GIT_COMMIT",
            "NODE_NAME",
            "GITHUB_WORKFLOW",
            "GITHUB_RUN_ID",
            "GITHUB_SHA",
            "LOCAL_RUN_ID",
        ]

        for var in all_vars:
            original_env[var] = os.getenv(var)
            if var in os.environ:
                del os.environ[var]

        yield

        for var, value in original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_jenkins_priority_over_github(self):
        """Test that Jenkins variables take priority over GitHub Actions"""
        # Set both Jenkins and GitHub variables
        os.environ["JOB_NAME"] = "jenkins-job"
        os.environ["BUILD_NUMBER"] = "5"
        os.environ["BUILD_ID"] = "2026-03-08_12-00-00"
        os.environ["GITHUB_WORKFLOW"] = "github-workflow"
        os.environ["GITHUB_RUN_ID"] = "999"

        workflow_name = get_git_workflow_name()
        run_id = get_git_run_id()

        # Jenkins should win
        assert workflow_name == "jenkins-job"
        assert run_id.startswith("jenkins_5_")

    def test_jenkins_commit_priority_over_github(self):
        """Test that Jenkins GIT_COMMIT takes priority over GITHUB_SHA"""
        os.environ["GIT_COMMIT"] = "jenkins_commit_abc"
        os.environ["GITHUB_SHA"] = "github_sha_xyz"

        commit_hash = get_commit_hash()

        # Jenkins should win
        assert commit_hash == "jenkins_commit_abc"

    def test_github_priority_over_local(self):
        """Test that GitHub Actions takes priority over local"""
        os.environ["GITHUB_WORKFLOW"] = "github-workflow"
        os.environ["GITHUB_RUN_ID"] = "888"

        workflow_name = get_git_workflow_name()
        run_id = get_git_run_id()

        # GitHub should win over local
        assert workflow_name == "github-workflow"
        assert run_id == "github_888"
        assert workflow_name != "local"

    def test_complete_priority_chain(self):
        """Test complete priority: Jenkins > GitHub > Local for all functions"""
        # Set all variables
        os.environ["JOB_NAME"] = "jenkins-job"
        os.environ["BUILD_NUMBER"] = "10"
        os.environ["BUILD_ID"] = "2026-03-08_15-00-00"
        os.environ["GIT_COMMIT"] = "jenkins_hash"
        os.environ["GITHUB_WORKFLOW"] = "github-workflow"
        os.environ["GITHUB_RUN_ID"] = "777"
        os.environ["GITHUB_SHA"] = "github_hash"

        context = get_git_context()

        # All should use Jenkins values
        assert context[GIT_WORKFLOW_NAME_ATTRIBUTE] == "jenkins-job"
        assert context[GIT_RUN_ID_ATTRIBUTE].startswith("jenkins_10_")
        assert context[GIT_COMMIT_HASH_ATTRIBUTE] == "jenkins_hash"


class TestGitBranchDetection:
    """Test branch name detection across CI/CD environments"""

    @pytest.fixture(autouse=True)
    def cleanup_env_vars(self):
        """Clean up environment variables before and after each test"""
        original_env = {}
        branch_vars = ["GIT_BRANCH", "GITHUB_HEAD_REF", "GITHUB_REF_NAME"]
        for var in branch_vars:
            original_env[var] = os.getenv(var)
            if var in os.environ:
                del os.environ[var]
        yield
        for var, value in original_env.items():
            if value is not None:
                os.environ[var] = value
            elif var in os.environ:
                del os.environ[var]

    def test_jenkins_branch_detection(self):
        """Test that Jenkins GIT_BRANCH is used"""
        os.environ["GIT_BRANCH"] = "feature/my-feature"
        assert get_git_branch() == "feature/my-feature"

    def test_github_head_ref_for_prs(self):
        """Test that GITHUB_HEAD_REF is used for pull requests"""
        os.environ["GITHUB_HEAD_REF"] = "feature/pr-branch"
        assert get_git_branch() == "feature/pr-branch"

    def test_github_ref_name_for_pushes(self):
        """Test that GITHUB_REF_NAME is used for push events"""
        os.environ["GITHUB_REF_NAME"] = "main"
        assert get_git_branch() == "main"

    def test_jenkins_branch_priority_over_github(self):
        """Test that Jenkins GIT_BRANCH takes priority over GitHub"""
        os.environ["GIT_BRANCH"] = "jenkins-branch"
        os.environ["GITHUB_HEAD_REF"] = "github-branch"
        assert get_git_branch() == "jenkins-branch"

    def test_github_head_ref_priority_over_ref_name(self):
        """Test that GITHUB_HEAD_REF takes priority over GITHUB_REF_NAME"""
        os.environ["GITHUB_HEAD_REF"] = "pr-branch"
        os.environ["GITHUB_REF_NAME"] = "main"
        assert get_git_branch() == "pr-branch"

    def test_branch_in_context(self):
        """Test that branch is included in get_git_context()"""
        os.environ["GIT_BRANCH"] = "release/1.0"
        context = get_git_context()
        assert GIT_BRANCH_ATTRIBUTE in context
        assert context[GIT_BRANCH_ATTRIBUTE] == "release/1.0"

    def test_branch_always_present_in_context(self):
        """Test that git.branch key is always present in context (even locally)"""
        context = get_git_context()
        assert GIT_BRANCH_ATTRIBUTE in context
        assert context[GIT_BRANCH_ATTRIBUTE]  # not empty


class TestGitRepoInContext:
    """Test that repo name is included in the git context"""

    def test_repo_attribute_key(self):
        """Test that GIT_REPO_ATTRIBUTE constant has expected value"""
        assert GIT_REPO_ATTRIBUTE == "git.repo"

    def test_repo_in_context_when_available(self):
        """Test that git.repo is present when a git repo is available"""
        context = get_git_context()
        # If running inside a git repo (which CI always is), repo should be present
        if GIT_REPO_ATTRIBUTE in context:
            assert context[GIT_REPO_ATTRIBUTE]  # not empty string
            assert "/" not in context[GIT_REPO_ATTRIBUTE]  # org-repo format uses "-"
