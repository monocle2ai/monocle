"""CrewAI instrumentation module for Monocle.

This module automatically disables CrewAI's built-in telemetry to avoid conflicts
with Monocle's telemetry instrumentation.
"""

import os

# Disable CrewAI's built-in telemetry to use Monocle's telemetry instead
# This prevents service name conflicts where CrewAI sets "crewAI-telemetry"
# instead of using the workflow name provided to Monocle
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
