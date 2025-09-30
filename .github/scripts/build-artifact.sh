#!/bin/bash

set -e

# Multi-artifact build script for Monocle
# Usage: ./build-artifact.sh [apptrace|mcp|test_tools] [version]

ARTIFACT=$1
VERSION=$2

if [[ -z "$ARTIFACT" ]]; then
    echo "Usage: $0 [apptrace|mcp|test_tools] [version]"
    echo "Example: $0 apptrace 0.5.4"
    exit 1
fi

# Validate artifact name
case $ARTIFACT in
    apptrace|mcp|test_tools)
        ;;
    *)
        echo "Error: Invalid artifact name '$ARTIFACT'. Must be one of: apptrace, mcp, test_tools"
        exit 1
        ;;
esac

# Validate artifact directory exists
if [[ ! -d "$ARTIFACT" ]]; then
    echo "Error: Directory '$ARTIFACT' does not exist"
    exit 1
fi

# Validate pyproject.toml exists
if [[ ! -f "$ARTIFACT/pyproject.toml" ]]; then
    echo "Error: pyproject.toml not found in '$ARTIFACT' directory"
    exit 1
fi

echo "🚀 Building artifact: $ARTIFACT"
cd "$ARTIFACT"

# Update version if provided
if [[ -n "$VERSION" ]]; then
    echo "📝 Updating version to: $VERSION"
    
    # Validate version format
    if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([ab][0-9]+)?$ ]]; then
        echo "Error: Invalid version format: $VERSION. Use formats like 0.5.0, 0.5.0a1, or 0.5.0b1"
        exit 1
    fi
    
    # Update version in pyproject.toml
    sed -i.bak "s/^version = .*/version = \"$VERSION\"/" pyproject.toml
    echo "✅ Updated version to $VERSION in $ARTIFACT/pyproject.toml"
    rm -f pyproject.toml.bak  # Clean up backup file
fi

# Get current version for display
CURRENT_VERSION=$(grep '^version = ' pyproject.toml | sed 's/version = "\(.*\)"/\1/')
echo "🔍 Building $ARTIFACT version: $CURRENT_VERSION"

# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Install build dependencies
echo "📦 Installing build dependencies..."
python3 -m pip install --upgrade pip build setuptools wheel

# Build the package
echo "⚙️ Building package..."
python3 -m build

# Display build results
echo "✅ Build completed successfully!"
echo "📁 Artifacts created in $ARTIFACT/dist/:"
ls -lh dist/

# Get package info
PACKAGE_NAME=$(grep '^name = ' pyproject.toml | sed 's/name = "\(.*\)"/\1/')
echo "📋 Package: $PACKAGE_NAME"
echo "📋 Version: $CURRENT_VERSION"
echo "📋 Location: $(pwd)/dist/"