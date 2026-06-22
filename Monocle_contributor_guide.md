# Monocle contributor guide
This document descripts process and best practices for contributing code/docs to Monocle

## Start with a fork of Monocle
It's best to make code change in your fork, on a separate branch
- Fork Monocle repo into your persona/org git organisation
- Sync the main in the forked repo with latest on Monocle main before you start making code changes.
- Create a separate branch in your fork for the pull request off the main.

## CDO requirement
All contributions to this project must be accompanied by acknowledgment of, and agreement to, the [Developer Certificate of Origin](https://github.com/apps/dco).
- Every commit in your branch needs to be signed (`git commit -s ...`)

## Testing
For any code/behavior changes, there should be tests
- If this is a code changes, then add a test cases that validates the changes.
- Run the existing [unit tests](#run-the-unit-tests) on your changes. If there are any regressions introduced with your change, address those in your patch before submitting.

## Documentation
- If the changes introduce any new behavior or change existing behavior, update the README and User guide to reflect new changes.

## Submit a git pull request
- Push your branch to github
- Create a Github pull request
- Link to the tracking issue if there's one
- Complete the pull request template
- Mentions any community members/maintainers to review your RP

## Stay engaged with Monocle community during the review process
- Respond to questions/feedback on the PR.
- Any changes you make to the origin PR should follow the [CDO requirements](#cdo-requirement) and [Testing](#testing)

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