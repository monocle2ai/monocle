TEST_SCOPE_NAME: str = "test_name"
GIT_RUN_ID_ATTRIBUTE: str = "git.run.id"
GIT_WORKFLOW_NAME_ATTRIBUTE: str = "git.workflow.name"
GIT_COMMIT_HASH_ATTRIBUTE: str = "git.commit.hash"
DEFAULT_WORKFLOW_NAME: str = "monocle_validator"
TEST_STATUS_ATTRIBUTE: str = "test.status"
TEST_ASSERTION_ATTRIBUTE: str = "test.assertion.message"

# Jenkins environment variables
JENKINS_BUILD_NUMBER: str = "BUILD_NUMBER"
JENKINS_BUILD_ID: str = "BUILD_ID"
JENKINS_JOB_NAME: str = "JOB_NAME"
JENKINS_BUILD_URL: str = "BUILD_URL"
JENKINS_URL: str = "JENKINS_URL"
JENKINS_NODE_NAME: str = "NODE_NAME"
JENKINS_GIT_COMMIT: str = "GIT_COMMIT"
JENKINS_GIT_BRANCH: str = "GIT_BRANCH"

# Jenkins trace attributes
JENKINS_URL_ATTRIBUTE: str = "jenkins.url"
JENKINS_BUILD_URL_ATTRIBUTE: str = "jenkins.build.url"
JENKINS_NODE_NAME_ATTRIBUTE: str = "jenkins.node.name"

# GitHub Actions environment variables (original)
GITHUB_WORKFLOW: str = "GITHUB_WORKFLOW"
GITHUB_RUN_ID: str = "GITHUB_RUN_ID"
GITHUB_SHA: str = "GITHUB_SHA"

# Local environment
LOCAL_RUN_ID: str = "LOCAL_RUN_ID"
