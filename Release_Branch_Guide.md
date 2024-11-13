### Guide: Preparing a Release Branch with GitHub Actions

This guide outlines how to set up a GitHub Actions workflow for preparing a release branch, including tasks like verifying prerequisites, creating pull requests, and updating the changelog. We'll focus on the steps and the relevant parts of the code for clarity.

---

### 1. **Workflow Trigger**
The workflow is manually triggered via the `workflow_dispatch` event. The key input here is the `prerelease_version` that specifies the version for a pre-release.

```yaml
on:
  workflow_dispatch:
    inputs:
      prerelease_version:
        description: "Pre-release version number"
        required: True
```

---

### 2. **Prerequisites Verification**

The first job, `prereqs`, verifies that the workflow is only triggered on the `main` branch and checks if the changelog contains an `Unreleased` section. It also verifies that the prerelease version is consistent with the stable version.

```yaml
if [[ $GITHUB_REF_NAME != main ]]; then
  echo this workflow should only be run against main
  exit 1
fi

if ! grep --quiet "^## Unreleased$" CHANGELOG.md; then
  echo the change log is missing an \"Unreleased\" section
  exit 1
fi
```

This ensures that the workflow is run only when appropriate, preventing errors during execution.

---

### 3. **Creating the Release Branch**

The next step in the `create-pull-request-against-release-branch` job is to generate a release branch name based on the stable and unstable versions. It uses a Python script (`eachdist.py`) to determine the version details and constructs the branch name.

```yaml
release_branch_name="release/v${stable_version_branch_part}-${unstable_version_branch_part}"
git push origin HEAD:$release_branch_name
```

This automatically creates a release branch, ready for further changes.

---

### 4. **Update the Changelog**

Once the release branch is created, the changelog is updated to reflect the pre-release version and its approximate release date.

```yaml
date=$(date "+%Y-%m-%d")
sed -Ei "s/^## Unreleased$/## Version ${{ github.event.inputs.prerelease_version }}\/ ($date)/" CHANGELOG.md
```

This ensures that the changelog reflects the changes and dates relevant to the pre-release.

---

### 5. **Creating Pull Request Against Release Branch**

A pull request is then created to merge the changes against the newly created release branch. The following code pushes the updates to a new branch and creates the pull request.

```yaml
git push origin HEAD:$branch
gh pr create --title "[$RELEASE_BRANCH_NAME] $message" --head $branch --base $RELEASE_BRANCH_NAME
```

This ensures that the changes are tracked in a pull request and can be reviewed before merging into the release branch.

---

### 6. **Creating Pull Request Against Main**

Finally, the `create-pull-request-against-main` job updates the `main` branch with the new version information and creates a pull request to update the `main` branch.

```yaml
git push origin HEAD:$branch
gh pr create --title "$message" --body "$body" --head $branch --base main
```

This is the final step to update `main` with the newly created release versions, ensuring that the project is always up-to-date.

---

