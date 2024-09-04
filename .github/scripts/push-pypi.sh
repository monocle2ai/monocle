python -m pip install pipenv
pipenv install build
pipenv install twine
source $(pipenv --venv)/bin/activate
python -m build
python -m twine upload --repository $PYPI_REPO dist/* --username $PYPI_USERNAME --password $PYPI_PASSWORD