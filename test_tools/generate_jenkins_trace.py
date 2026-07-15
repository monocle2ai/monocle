#!/usr/bin/env python3
"""Generate a sample trace with Jenkins metadata for demonstration."""

import os
import sys

# Set Jenkins environment variables
os.environ["JENKINS_URL"] = "https://jenkins.example.com"
os.environ["BUILD_NUMBER"] = "123"
os.environ["BUILD_ID"] = "2026-03-09_13-30-45"
os.environ["JOB_NAME"] = "monocle-test-pipeline"
os.environ["BUILD_URL"] = "https://jenkins.example.com/job/monocle-test-pipeline/123/"
os.environ["NODE_NAME"] = "jenkins-agent-001"
os.environ["GIT_COMMIT"] = "abc123def456789"

# Simple test case
from monocle_test_tools.validator import MonocleValidator
from monocle_test_tools.schema import TestCase

# Initialize validator
validator = MonocleValidator()

# Create a simple test case
test_case = TestCase(
    test_name="jenkins_demo_test",
    test_case_name="Jenkins Demo Test",
    test_input=["Hello from Jenkins!"],
    test_output="Demo output",
)

# Start the test run
token = validator.pre_test_run_setup("jenkins_demo_test", None)

print("Generated trace with Jenkins context:")
print("=" * 60)

# Get the git context to show what was captured
from monocle_test_tools.gitutils import get_git_context
context = get_git_context()

for key, value in sorted(context.items()):
    print(f"  {key}: {value}")

print("=" * 60)

# Clean up
validator.post_test_cleanup(token, "jenkins_demo_test", False)

print("\nTrace file saved to: .monocle/test_traces/")
print("Check the latest JSON file for Jenkins metadata!")
