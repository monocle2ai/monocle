"""pytest options for the parametrized CSV example (csv_cases_parametrized_example.py).

pytest only invokes pytest_addoption from a conftest/plugin, so the example's
--fact-set / --template flags are declared here. These are inert for every other
test (just unused options).
"""


def pytest_addoption(parser):
    group = parser.getgroup("csv eval example")
    group.addoption("--fact-set", action="store", default=None,
                    help="Path to the fact-set CSV for the parametrized example.")
    group.addoption("--template", action="store", default=None,
                    help="Custom template path (or built-in template name) for the example.")
