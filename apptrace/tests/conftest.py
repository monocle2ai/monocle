import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load the rerun plugin
pytest_plugins = [
    "config.rerun_test.plugin",
]
