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
> pipenv install monocle-apptrace
```
### Running the testcases

 
##### Activate the virtual environment
```
cd monocle
monocle% python -m pip install pipenv
monocle% pipenv --python 3.11.9
monocle% source $(pipenv --venv)/bin/activate

 ```
##### Install the dependencies
```
monocle% pip install -e '.[dev]'
 ```

##### Run the unit tests
```
monocle% pytest tests/unit/*_test.py
 ```
 
##### Run the integration tests
```
monocle% pytest -m integration
 ```

###### Run the integration test individually
```
monocle% pytest tests/integration/test_langchain_rag_l_to_m.py
```