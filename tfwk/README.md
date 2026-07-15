# Monocle Test Framework (tfwk)

A pytest-style testing framework for validating AI agent behavior through trace analysis.

## Overview

The monocle test framework provides an intuitive way to write tests for AI agents by analyzing their execution traces. Inspired by [AgentiTest](https://github.com/kweinmeister/agentitest), this framework uses pytest conventions with fluent assertions for trace validation.

## Installation

```bash
# From the tfwk directory
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
import pytest
from monocle_tfwk import BaseAgentTest

class TestMyAgent(BaseAgentTest):
    def test_basic_response(self):
        # Your agent code here
        result = my_agent.run("What is 2+2?")
        
        # Fluent assertions on traces
        (self.assert_traces()
         .has_spans(min_count=1)
         .has_llm_calls(count=1)
         .llm_calls()
         .contains_input("What is 2+2?")
         .contains_output("4"))
```

## Key Features

- **Pytest Integration**: Works seamlessly with pytest fixtures and conventions
- **Fluent Assertions**: Chainable methods for readable test code
- **Trace Analysis**: Built-in support for OpenTelemetry span analysis
- **Common Patterns**: Helper methods for frequent testing scenarios
- **Async Support**: Full support for async agent testing

## Documentation

See the `examples/` directory for comprehensive usage examples and patterns.

## License

Licensed under the Apache License 2.0 - see LICENSE file for details.