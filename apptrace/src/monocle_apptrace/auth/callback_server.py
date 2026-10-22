# Old import path, kept for one release.
from monocle_apptrace.providers.moved import alias_module

alias_module(__name__, "monocle_apptrace.providers.okahu.auth.callback_server")
