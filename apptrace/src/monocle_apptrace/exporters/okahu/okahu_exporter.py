# Old import path, kept for one release.
import sys
import warnings

from monocle_apptrace.providers.okahu import okahu_exporter as _moved

warnings.warn(
    "monocle_apptrace.exporters.okahu.okahu_exporter has moved to "
    "monocle_apptrace.providers.okahu.okahu_exporter",
    DeprecationWarning, stacklevel=2)

# Point the old name at the moved module, so imports and mock.patch both still work.
sys.modules[__name__] = _moved
