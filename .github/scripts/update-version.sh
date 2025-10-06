#!/bin/bash -e

# Multi-artifact version update script for Monocle
# Usage: ./update-version.sh <new_version> [artifact_name]
# artifact_name: apptrace, mcp, test_tools, all (defaults to apptrace)

# Check if version parameter is provided
if [ -z "$1" ]; then
    echo "Error: Version parameter is required"
    echo "Usage: ./update-version.sh <new_version> [artifact_name]"
    echo "Example: ./update-version.sh 1.2.3 apptrace"
    echo "Example: ./update-version.sh 1.2.3 all"
    echo "Artifacts: apptrace, mcp, test_tools, all"
    exit 1
fi

NEW_VERSION=$1
ARTIFACT=${2:-"apptrace"}

# Function to update version in a specific artifact
update_artifact_version() {
    local artifact=$1
    local version=$2
    
    # Validate artifact directory exists
    if [[ ! -d "$artifact" ]]; then
        echo "Error: Directory '$artifact' does not exist"
        return 1
    fi

    # Validate pyproject.toml exists  
    if [[ ! -f "$artifact/pyproject.toml" ]]; then
        echo "Error: pyproject.toml not found in '$artifact' directory"
        return 1
    fi

    # Update version
    sed -i "s/^version =.*/version = \"$version\"/" "$artifact/pyproject.toml"
    
    if [ $? -eq 0 ]; then
        echo "Successfully updated $artifact version to $version"
        return 0
    else
        echo "Error: Failed to update $artifact version"
        return 1
    fi
}

# Validate version format (basic check for x.y.z pattern)
if [[ ! $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+([ab][0-9]+|rc[0-9]+)?$ ]]; then
    echo "Error: Invalid version format. Must be like X.Y.Z, X.Y.Za1, X.Y.Zb1, or X.Y.Zrc1"
    exit 1
fi

# Handle different artifact options
case $ARTIFACT in
    apptrace|mcp|test_tools)
        update_artifact_version "$ARTIFACT" "$NEW_VERSION"
        ;;
    all)
        echo "Updating all artifacts to version $NEW_VERSION..."
        failed=0
        for art in apptrace mcp test_tools; do
            if ! update_artifact_version "$art" "$NEW_VERSION"; then
                failed=1
            fi
        done
        if [ $failed -eq 1 ]; then
            echo "Error: Some artifacts failed to update"
            exit 1
        fi
        echo "All artifacts successfully updated to $NEW_VERSION"
        ;;
    *)
        echo "Error: Invalid artifact name '$ARTIFACT'. Must be one of: apptrace, mcp, test_tools, all"
        exit 1
        ;;
esac