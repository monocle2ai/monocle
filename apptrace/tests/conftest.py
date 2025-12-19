"""
Root conftest for apptrace tests.
Loads plugins that need to be available during option parsing.
"""
import sys
from pathlib import Path

# Ensure config.rerun_test is importable
sys.path.insert(0, str(Path(__file__).parent))

# Load the rerun plugin
pytest_plugins = [
    "config.rerun_test.plugin",
]
