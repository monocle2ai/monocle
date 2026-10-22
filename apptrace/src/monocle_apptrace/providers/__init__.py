"""Providers: named backends that Monocle talks to. Core code finds them by name."""
from monocle_apptrace.providers.base import Provider
from monocle_apptrace.providers.registry import NamedRegistry, get_provider, provider_names, register_provider

__all__ = ["NamedRegistry", "Provider", "get_provider", "provider_names", "register_provider"]
