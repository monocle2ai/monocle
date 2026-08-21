"""Monocle span exporters and filtering utilities."""

from monocle_apptrace.exporters.span_filter import (
    SpanFilter,
    FilteredSpanExporter,
)
from monocle_apptrace.exporters.monocle_exporters import (
    get_monocle_exporter,
    monocle_exporters,
)
from monocle_apptrace.exporters.base_exporter import (
    SpanExporterBase,
    MonocleInMemorySpanExporter,
)
from monocle_apptrace.exporters.span_obfuscator import (
    SpanObfuscator,
    TextSpanObfuscator,
    RegexSpanObfuscator,
    PresidioSpanObfuscator,
    ObfuscatingSpanExporter,
    ObfuscatingSpanProcessor,
    register_span_obfuscator,
    set_span_obfuscators,
    get_span_obfuscators,
    obfuscation_disabled_by_env,
    wrap_exporter_with_obfuscation,
    install_obfuscation_hook,
    install_obfuscation_hooks,
)

__all__ = [
    # Filtering
    "SpanFilter",
    "FilteredSpanExporter",

    # Sensitive data obfuscation
    "SpanObfuscator",
    "TextSpanObfuscator",
    "RegexSpanObfuscator",
    "PresidioSpanObfuscator",
    "ObfuscatingSpanExporter",
    "ObfuscatingSpanProcessor",
    "register_span_obfuscator",
    "set_span_obfuscators",
    "get_span_obfuscators",
    "obfuscation_disabled_by_env",
    "wrap_exporter_with_obfuscation",
    "install_obfuscation_hook",
    "install_obfuscation_hooks",


    # Exporter registry
    "get_monocle_exporter",
    "monocle_exporters",
    
    # Base classes
    "SpanExporterBase",
    "MonocleInMemorySpanExporter",
]
