# load the plugin for local test runs without installing the package
# pytest_plugins = ["monocle_test_tools.pytest_plugin"]
# Load the plugin manually only when the package is NOT installed (e.g. local
# runs that rely on `pythonpath = ["src", ...]`). When the package IS installed,
# its `pytest11` entry point already registers this plugin; registering it again
# here would raise "Plugin already registered under a different name".
from importlib.metadata import entry_points as _entry_points


def _plugin_registered_via_entrypoint() -> bool:
    eps = _entry_points()
    # Python 3.10+ exposes the selectable API; 3.8/3.9 return a dict.
    group = eps.select(group="pytest11") if hasattr(eps, "select") else eps.get("pytest11", [])
    return any(ep.value == "monocle_test_tools.pytest_plugin" for ep in group)


if not _plugin_registered_via_entrypoint():
    pytest_plugins = ["monocle_test_tools.pytest_plugin"]
