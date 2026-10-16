"""Find providers by name.

A provider comes from one of three places, checked in this order:
  1. register_provider() at runtime
  2. the providers that ship with Monocle (_BUILTIN_PROVIDERS)
  3. installed packages that declare a "monocle.providers" entry point
A provider module is imported only when its name is asked for.
"""
from importlib import import_module
from typing import Dict, List, Optional

from monocle_apptrace.providers.base import Provider

ENTRY_POINT_GROUP = "monocle.providers"

# Providers that ship with Monocle, as "module:Class".
_BUILTIN_PROVIDERS: Dict[str, str] = {
    "okahu": "monocle_apptrace.providers.okahu.provider:OkahuProvider",
}

_providers: Dict[str, Provider] = {}


def register_provider(provider: Provider) -> None:
    """Add a provider, or replace the one with the same name."""
    _providers[provider.name] = provider


def get_provider(name: str) -> Optional[Provider]:
    """Return the provider with this name, or None if there is none."""
    if name in _providers:
        return _providers[name]
    target = _BUILTIN_PROVIDERS.get(name) or _entry_point_targets().get(name)
    if target is None:
        return None
    module_name, _, class_name = target.partition(":")
    provider = getattr(import_module(module_name), class_name)()
    _providers[name] = provider
    return provider


def provider_names() -> List[str]:
    """Names of all known providers, loaded or not."""
    return sorted(set(_providers) | set(_BUILTIN_PROVIDERS) | set(_entry_point_targets()))


def _entry_point_targets() -> Dict[str, str]:
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return {}
    eps = entry_points()
    # Python 3.10+ has select(); 3.8/3.9 return a dict of groups.
    group = eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, "select") else eps.get(ENTRY_POINT_GROUP, [])
    return {ep.name: ep.value for ep in group}
