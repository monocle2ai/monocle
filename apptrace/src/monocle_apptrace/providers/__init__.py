"""Providers: named backends that Monocle talks to. Core code finds them by name."""
from monocle_apptrace.providers.base import Provider
from monocle_apptrace.providers.registry import get_provider, provider_names, register_provider

__all__ = ["Provider", "get_provider", "provider_names", "register_provider"]
