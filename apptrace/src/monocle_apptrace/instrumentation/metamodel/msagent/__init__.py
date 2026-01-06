"""Microsoft Agent Framework instrumentation module."""

from monocle_apptrace.instrumentation.metamodel.msagent.methods import MSAGENT_METHODS
from monocle_apptrace.instrumentation.metamodel.msagent.msagent_processor import (
    MSAgentRequestHandler,
    MSAgentAgentHandler,
    MSAgentToolHandler,
)

__all__ = [
    "MSAGENT_METHODS",
    "MSAgentRequestHandler",
    "MSAgentAgentHandler",
    "MSAgentToolHandler",
]
