# Monocle release process
## Propose a release
- Any contributor or commiter can propose a release by creating a ticket/issue
- Any TSC member can approve and initiate the release process

## Release steps
- Create a release branch off main
  - Branch name should be release/<version-number>
- Update the project.toml files to update the new artifact versions
- Update CHANGELOG.md to list out the changes/RPs included in the release
- Commit the changes to release branch
- Create a tag for the top of the release branch and sign it
  - `git tag -s v-<rel-number> -m "Release <rel-number>"`
  - `git push origin v-<rel-number>`
- Create PR for release
- Request other TSC members to approve the PR
- Execute the github release action to publish the release off the signed tag
