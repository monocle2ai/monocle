#!/bin/bash

# run_parallel_tests.sh - Script to run all unit tests in parallel
# Created: May 9, 2025

# Set up environment variables

# Determine the number of CPU cores for optimal parallelization
NUM_CORES=$(sysctl -n hw.ncpu)
# Use one less than the total number of cores to avoid overwhelming the system
PARALLEL_JOBS=$((NUM_CORES - 1))
# Ensure at least 2 parallel jobs
if [ $PARALLEL_JOBS -lt 2 ]; then
  PARALLEL_JOBS=2
fi

echo "Starting parallel unit tests with $PARALLEL_JOBS workers..."

# Check if pytest-xdist is installed, install if not
if ! python -c "import xdist" &> /dev/null; then
  echo "Installing pytest-xdist for parallel testing..."
  pip install pytest-xdist
fi

# Run unit tests in parallel
# -v: verbose output
# -n: number of parallel workers
# --no-header: suppress header
# --no-summary: suppress summary
# -xvs: show output, be verbose, and don't capture stdout
# tests/unit: directory containing unit tests
python -m pytest tests/unit -v -n $PARALLEL_JOBS -xvs

# Store the exit code
EXIT_CODE=$?

echo "All unit tests completed with exit code: $EXIT_CODE"
exit $EXIT_CODE