# Tests for monocle_tfwk

import os
from pathlib import Path

# Set up trace output path for integration tests
tfwk_root = Path(__file__).parent.parent.parent  # Navigate to tfwk directory
trace_output_path = tfwk_root / ".monocle"

# Ensure the trace output directory exists
trace_output_path.mkdir(exist_ok=True)

# Set the environment variable for all integration tests
os.environ["MONOCLE_TRACE_OUTPUT_PATH"] = str(trace_output_path)