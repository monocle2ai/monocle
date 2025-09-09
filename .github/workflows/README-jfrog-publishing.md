# JFrog Publishing Workflow

This GitHub Action workflow template enables publishing Python packages to JFrog Artifactory repositories.

## Features

- **Optional Branch Selection**: Choose which branch to build from (defaults to `main`)
- **Version Management**: Specify package version with support for alpha/beta releases
- **JFrog Integration**: Publishes to JFrog Artifactory using repository URL and credentials from GitHub secrets
- **Validation**: Validates version format and ensures required secrets are present
- **Build Process**: Uses the existing build script and Python 3.13 environment

## Usage

### Manual Trigger

1. Go to the **Actions** tab in your GitHub repository
2. Select **"Publish to JFrog"** workflow
3. Click **"Run workflow"**
4. Fill in the parameters:
   - **Branch** (optional): The branch to build from. Defaults to `main` if not specified
   - **Version** (required): The version to release (e.g., `1.0.0`, `1.0.0-alpha`, `1.0.0-beta1`)

### Required Secrets

Configure these secrets in your repository settings (**Settings** → **Secrets and variables** → **Actions**):

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `JFROG_REPOSITORY_URL` | JFrog Artifactory repository URL | `https://company.jfrog.io/artifactory/api/pypi/pypi-local` |
| `JFROG_USERNAME` | JFrog username for authentication | `your-username` |
| `JFROG_PASSWORD` | JFrog password or API token | `your-password-or-token` |

### Version Format

The workflow accepts the following version formats:

- **Release versions**: `1.0.0`, `2.1.3`, `0.5.0`
- **Alpha versions**: `1.0.0-alpha`, `1.0.0-alpha1`, `1.0.0-alpha123`
- **Beta versions**: `1.0.0-beta`, `1.0.0-beta1`, `1.0.0-beta99`

### Environment

The workflow runs in the `ArtifactPublish` environment, which may require additional approvals or protection rules depending on your repository configuration.

## Workflow Steps

1. **Checkout**: Checks out the specified branch (or `main` if not specified)
2. **Version Validation**: Validates the provided version format
3. **Version Update**: Updates the version in `pyproject.toml`
4. **Python Setup**: Sets up Python 3.13 environment
5. **Build**: Executes the build script to create distribution packages
6. **Install Twine**: Installs the twine package for uploading
7. **Publish**: Uploads the built packages to JFrog Artifactory
8. **Summary**: Provides a summary of the successful publication

## Example Usage

```yaml
# Publish version 1.2.0 from main branch
inputs:
  branch: main  # optional, defaults to main
  version: 1.2.0

# Publish alpha version from develop branch  
inputs:
  branch: develop
  version: 1.3.0-alpha1

# Publish beta version (branch defaults to main)
inputs:
  version: 1.2.1-beta
```

## Error Handling

The workflow includes validation and error handling for:

- Invalid version formats
- Missing required secrets
- Build failures
- Upload failures

If any step fails, the workflow will stop and provide error details in the logs.