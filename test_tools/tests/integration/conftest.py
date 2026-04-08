# load the plugin for local test runs without installing the package
# Only register if not already loaded via the installed package's entry point
import importlib.util as _iu
if _iu.find_spec("monocle_test_tools") is None:
    pytest_plugins = ["monocle_test_tools.pytest_plugin"]
