#!/bin/bash

# Exit on error
set -e

# Multi-artifact version script for Monocle
# Usage: ./get-version.sh [artifact_name]
# artifact_name: apptrace, mcp, test_tools (defaults to apptrace)

ARTIFACT=${1:-"apptrace"}

# Validate artifact name
case $ARTIFACT in
    apptrace|mcp|test_tools)
        ;;
    *)
        echo "Error: Invalid artifact name '$ARTIFACT'. Must be one of: apptrace, mcp, test_tools" >&2
        exit 1
        ;;
esac

# Validate artifact directory exists
if [[ ! -d "$ARTIFACT" ]]; then
    echo "Error: Directory '$ARTIFACT' does not exist" >&2
    exit 1
fi

# Validate pyproject.toml exists
if [[ ! -f "$ARTIFACT/pyproject.toml" ]]; then
    echo "Error: pyproject.toml not found in '$ARTIFACT' directory" >&2
    exit 1
fi

# Extract version using grep and sed
# Find line with version =, extract content between quotes
version=$(grep '^version = ' "$ARTIFACT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/')

# Print version number only
echo "$version"