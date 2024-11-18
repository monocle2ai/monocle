#!/bin/bash

# Exit on error
set -e

# Extract version using grep and sed
# Find line with version =, extract content between quotes
version=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')

# Print version number only
echo "$version"