"""Validators imported by path from test_check_validator.py.

A module of their own because that is what a JSON test case must name: JSON
cannot hold a function.
"""
from monocle_test_tools import BaseValidator


def accepts_all(input=None, output=None):
    return True


def rejects_all(input=None, output=None):
    return "rejected by import-path validator"


class MinimumLengthValidator(BaseValidator):
    """A validator with options, covering the class form."""

    def validate_response(self, input=None, output=None):
        minimum = self.validator_options.get("minimum", 1)
        if output is None or len(output) < minimum:
            return f"output shorter than {minimum} characters"
        return True


not_callable = "this is a string, not a validator"
