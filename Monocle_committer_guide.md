# Monocle Committer Guide
This document provide details for Monocle committers tasks

## Build and publishing python packages
### Building the package

```
> python3 -m build 
```
### Publishing the package

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
> pipenv install monocle-observability
```
