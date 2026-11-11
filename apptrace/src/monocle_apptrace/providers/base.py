from typing import Dict, Optional, Tuple


class Provider:
    """A named backend that Monocle can send data to or read data from.

    A subclass sets ``name`` and fills in only the hooks it supports.
    """

    name: str = ""

    # Span exporter as {"module": ..., "class": ...}, same shape as the
    # entries in monocle_exporters. None if the provider has no exporter.
    span_exporter: Optional[Dict[str, str]] = None

    def get_setup_env(self, args) -> Tuple[Optional[Dict[str, str]], bool]:
        """This provider's step in ``monocle-apptrace <agent>-setup``.

        Returns ``(env, aborted)``: the settings to save, or None to use local
        files; and True if the user tried to set it up and it failed.
        """
        return None, False
