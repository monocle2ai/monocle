# Release Process Documentation

This document explains the automated release workflows for the Monocle project.

## Overview

The release process is handled by two GitHub workflows:

1. `prepare-release-branch.yml` - Creates release branches and prepares version updates
2. `release.yml` - Handles the actual release process including publishing to PyPI

## Prepare Release Branch Workflow

### Trigger
- Manual trigger via GitHub Actions UI
- Optional input: `prerelease_version` for pre-releases (e.g. "1.9.0rc2")

### Steps

1. **Prerequisites Check**
   - Verifies workflow is run against main branch
   - Checks for "Unreleased" section in CHANGELOG.md
   - For pre-releases, validates version matches main branch version

2. **Create Release Branch**
   - For regular releases (X.Y.0):
     - Creates branch `release/vX.Y.0`
   - For pre-releases (X.Y.0rcZ):
     - Creates branch `release/vX.Y.0rcZ`

3. **Update Release Branch**
   - Updates CHANGELOG.md with release date
   - Creates PR against release branch with version updates
   
4. **Update Main Branch**
   - Increments version to next minor version (X.Y+1.0)
   - Updates CHANGELOG.md
   - Creates PR against main branch with version bump

## Release Workflow 

### Trigger
- Manual trigger via GitHub Actions UI
- Must be run against a release/* branch

### Steps

1. **Version Validation**
   - Extracts version from pyproject.toml
   - Handles patch releases by tracking previous version

2. **Build & Publish**
   - Builds Python package
   - Publishes to TestPyPI first as validation
   - Can optionally publish to PyPI (currently commented out)

3. **GitHub Release**
   - Generates release notes from CHANGELOG.md
   - Creates GitHub release with generated notes
   - Tags release with version number
   - Creates release discussion

4. **Sync Changes**
   - Updates main branch CHANGELOG.md with release date
   - For non-patch releases:
     - Updates existing version entry
   - For patch releases:
     - Adds new version entry
   - Creates PR to sync changes back to main

## Version Numbering

- Regular releases use X.Y.0 format
- Pre-releases use X.Y.0rcZ format 
- Patch releases use X.Y.Z format where Z > 0

## Files Modified

- CHANGELOG.md - Release notes and dates
- pyproject.toml - Version number
- Git tags - Release tags

## Pull Requests Created

1. Against release branch:
   - Title: "Prepare release X.Y.0"
   - Updates version and changelog

2. Against main branch:
   - For initial release:
     - Title: "Update version to X.Y+1.0" 
     - Bumps version number
   - After release:
     - Title: "Copy change log updates from release/vX.Y.0"
     - Syncs changelog changes