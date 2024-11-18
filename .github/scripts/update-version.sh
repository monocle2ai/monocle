#!/bin/bash -e

# Updates version in pyproject.toml

# Check if version parameter is provided
if [ -z "$1" ]; then
    echo "Error: Version parameter is required"
    echo "Usage: ./update-version.sh <new_version>"
    echo "Example: ./update-version.sh 1.2.3"
    exit 1
fi

NEW_VERSION=$1

# # Validate version format (basic check for x.y.z pattern)
# if [[ ! $NEW_VERSION =~ ^[0-9]+\.[0-9]+\.[0-9]+(-?(dev|rc|alpha|beta)[0-9]*)?$ ]]; then
#     echo "Error: Invalid version format. Must be like X.Y.Z or X.Y.Z-dev"
#     exit 1
# fi


sed -i "s/^version =.*/version = \"$NEW_VERSION\"/" pyproject.toml


if [ $? -eq 0 ]; then
    echo "Successfully updated version to $NEW_VERSION"
else
    echo "Error: Failed to update version"
    exit 1
fi