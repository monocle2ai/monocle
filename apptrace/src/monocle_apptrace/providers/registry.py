from importlib import import_module
from typing import Any, Dict, List, Optional

from monocle_apptrace.providers.base import Provider


class NamedRegistry:
    """Objects found by name.

    An object comes from one of three places, checked in this order:
      1. register() at runtime
      2. the built-in table, as "module:attr"
      3. installed packages that declare an entry point in this group
    A built-in or entry-point object is imported only when its name is asked for.
    """

    def __init__(self, entry_point_group: str, builtins: Dict[str, str]):
        self.entry_point_group = entry_point_group
        self.builtins = builtins
        self._loaded: Dict[str, Any] = {}

    def register(self, name: str, obj: Any) -> None:
        """Add an object, or replace the one with the same name."""
        self._loaded[name] = obj

    def get(self, name: str) -> Optional[Any]:
        """Return the object with this name, or None if there is none."""
        if name in self._loaded:
            return self._loaded[name]
        target = self.builtins.get(name) or self._entry_point_targets().get(name)
        if target is None:
            return None
        module_name, _, attr = target.partition(":")
        obj = getattr(import_module(module_name), attr)
        self._loaded[name] = obj
        return obj

    def names(self) -> List[str]:
        """Names of all known objects, loaded or not."""
        return sorted(set(self._loaded) | set(self.builtins) | set(self._entry_point_targets()))

    def _entry_point_targets(self) -> Dict[str, str]:
        try:
            from importlib.metadata import entry_points
        except ImportError:
            return {}
        eps = entry_points()
        # Python 3.10+ has select(); 3.8/3.9 return a dict of groups.
        if hasattr(eps, "select"):
            group = eps.select(group=self.entry_point_group)
        else:
            group = eps.get(self.entry_point_group, [])
        return {ep.name: ep.value for ep in group}


_providers = NamedRegistry("monocle.providers", {
    "okahu": "monocle_apptrace.providers.okahu.provider:OkahuProvider",
})


def register_provider(provider: Provider) -> None:
    """Add a provider, or replace the one with the same name."""
    _providers.register(provider.name, provider)


def get_provider(name: str) -> Optional[Provider]:
    """Return the provider with this name, or None if there is none."""
    provider = _providers.get(name)
    # Built-in and entry-point targets name a class; keep one instance.
    if isinstance(provider, type):
        provider = provider()
        _providers.register(name, provider)
    return provider


def provider_names() -> List[str]:
    """Names of all known providers, loaded or not."""
    return _providers.names()
