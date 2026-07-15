import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Load the test repeat and pass rate tracking plugin
pytest_plugins = [
    "config.repeat_test.plugin",
]
