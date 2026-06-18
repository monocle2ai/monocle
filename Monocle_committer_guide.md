# Monocle Committer Guide
This document provide details for Monocle committers tasks

## Build and publishing python packages
### Prepare release
- Create a release branch off main
  - The branch name should be release/<release-version>
- Update the version number in pyprojects.toml in the artifacts being release (monocle_apptrace, monocle_test_tools, monocle_mcp)
- [Run](Monocle_contributor_guide.md#testing) the unit and integration tests
- If there are test failures, log a ticket to track it
- Triage the test failures
  - If this is a regression, then it should be fixed. A new release shouldn't be published with know regressions
  - If this a new feature failures, then community can make the call to release with know limitation documented in the Git release notes    
- Update the changelog to mention the highlights of the release
- Push the release branch

### Release via Git actions
This is the recommended process for Monocle releases
- To release new Monocle version, run the [Git Release Action](https://github.com/monocle2ai/monocle/actions/workflows/release.yml)
- The action should be run on a release branch or tag
- The action can be run by a maintainer and to be approved by another maintainer

### Manual release process (not recommended)
#### Building the package

```
> python3 -m build 
```
#### Publishing the package

```
> python3 -m pip install --upgrade twine
> python3 -m twine upload --repository testpypi dist/*
```

### Installing the package

The steps to set the credential can be found here:
https://packaging.python.org/en/latest/specifications/pypirc/

After setup of credentials, follow the commands below to publish the package to testpypi:

```
> python3 -m pip install pipenv
> pipenv install monocle-apptrace
```
