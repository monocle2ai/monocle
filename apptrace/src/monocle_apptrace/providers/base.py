from typing import Dict, Optional


class Provider:
    """A named backend that Monocle can send data to or read data from.

    A subclass sets ``name`` and fills in only the hooks it supports.
    """

    name: str = ""

    # Span exporter as {"module": ..., "class": ...}, same shape as the
    # entries in monocle_exporters. None if the provider has no exporter.
    span_exporter: Optional[Dict[str, str]] = None
