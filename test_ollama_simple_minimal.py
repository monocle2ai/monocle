"""
Minimal Ollama instrumentation demo with console exporter.
No assertions, just shows telemetry output.
"""

import sys
import os
# Add src directory to Python path to import monocle_apptrace directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from monocle_apptrace.instrumentation.common.instrumentor import setup_monocle_telemetry

setup_monocle_telemetry(
    workflow_name="ollama_simple_test",
    # monocle_exporters_list='console,file'
)

from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='gemma3:4b', messages=[
  {
    'role': 'system',
    'content': 'You are an AI assistant, give concise answers.',
  },
  {
    'role': 'user',
    'content': 'Why is the sky blue?',
  },
])
print(response['message']['content'])

