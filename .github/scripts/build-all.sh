#!/bin/bash

set -e

# Build multiple artifacts for Monocle
# Usage: ./build-all.sh [artifacts] [--apptrace-version VERSION] [--mcp-version VERSION] [--test-tools-version VERSION]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default values
ARTIFACTS="all"
APPTRACE_VERSION=""
MCP_VERSION=""
TEST_TOOLS_VERSION=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --apptrace-version)
            APPTRACE_VERSION="$2"
            shift 2
            ;;
        --mcp-version)
            MCP_VERSION="$2"
            shift 2
            ;;
        --test-tools-version)
            TEST_TOOLS_VERSION="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [artifacts] [options]"
            echo ""
            echo "Artifacts (default: all):"
            echo "  all              Build all artifacts"
            echo "  apptrace         Build only monocle_apptrace"
            echo "  mcp              Build only monocle_mcp"
            echo "  test_tools       Build only monocle_test_tools"
            echo "  apptrace,mcp     Build multiple specific artifacts"
            echo ""
            echo "Options:"
            echo "  --apptrace-version VERSION    Set version for apptrace"
            echo "  --mcp-version VERSION         Set version for mcp"
            echo "  --test-tools-version VERSION  Set version for test_tools"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Build all with current versions"
            echo "  $0 apptrace --apptrace-version 0.5.4 # Build only apptrace with new version"
            echo "  $0 apptrace,mcp                      # Build only apptrace and mcp"
            exit 0
            ;;
        *)
            if [[ -z "$ARTIFACTS" || "$ARTIFACTS" == "all" ]]; then
                ARTIFACTS="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

cd "$ROOT_DIR"

echo "🏗️ Monocle Multi-Artifact Build Script"
echo "=====================================\n"

# Parse which artifacts to build
BUILD_APPTRACE=false
BUILD_MCP=false
BUILD_TEST_TOOLS=false

if [[ "$ARTIFACTS" == "all" ]]; then
    BUILD_APPTRACE=true
    BUILD_MCP=true
    BUILD_TEST_TOOLS=true
    echo "📦 Building all artifacts"
else
    [[ "$ARTIFACTS" == *"apptrace"* ]] && BUILD_APPTRACE=true
    [[ "$ARTIFACTS" == *"mcp"* ]] && BUILD_MCP=true
    [[ "$ARTIFACTS" == *"test_tools"* ]] && BUILD_TEST_TOOLS=true
    echo "📦 Building selected artifacts: $ARTIFACTS"
fi

echo ""

# Build apptrace
if [[ "$BUILD_APPTRACE" == "true" ]]; then
    echo "🔧 Building monocle_apptrace..."
    ./.github/scripts/build-artifact.sh apptrace "$APPTRACE_VERSION"
    echo ""
fi

# Build mcp
if [[ "$BUILD_MCP" == "true" ]]; then
    echo "🔧 Building monocle_mcp..."
    ./.github/scripts/build-artifact.sh mcp "$MCP_VERSION"
    echo ""
fi

# Build test_tools
if [[ "$BUILD_TEST_TOOLS" == "true" ]]; then
    echo "🔧 Building monocle_test_tools..."
    ./.github/scripts/build-artifact.sh test_tools "$TEST_TOOLS_VERSION"
    echo ""
fi

echo "🎉 All builds completed successfully!"
echo ""
echo "📁 Build artifacts are located in:"
[[ "$BUILD_APPTRACE" == "true" ]] && echo "  - apptrace/dist/"
[[ "$BUILD_MCP" == "true" ]] && echo "  - mcp/dist/"
[[ "$BUILD_TEST_TOOLS" == "true" ]] && echo "  - test_tools/dist/"