import os
import pytest
from monocle_test_tools import MonocleValidator
from test_common.adk_travel_agent_litellm_openai import root_agent

agent_test_cases = [
    {
        "test_input": ["Book a flight from San Jose to Seattle for 27th Nov 2025."],
        "test_spans": [
            {
                "span_type": "agentic.turn",
                "eval": {
                    "eval": "regex_match",
                    "eval_options": {"pattern": r"seattle|san\s+jose", "ignore_case": True},
                    "args": ["output"],
                    "expected_result": {"match": 1.0},
                    "comparer": "metric",
                },
            },
            {
                "span_type": "agentic.turn",
                "eval": {
                    "eval": "json_validity",
                    "args": ["output"],
                    "expected_result": {"valid_json": 0.0},
                    "comparer": "metric",
                },
            },
            {
                "span_type": "agentic.turn",
                "eval": {
                    "eval": "keyword_presence",
                    "eval_options": {
                        "required_keywords": ["flight"],
                        "forbidden_keywords": ["password", "traceback"],
                    },
                    "args": ["output"],
                    "expected_result": {
                        "required_coverage": 1.0,
                        "forbidden_absent": 1.0,
                    },
                    "comparer": "metric",
                },
            },
            {
                "span_type": "agentic.turn",
                "eval": {
                    "eval": "exact_match",
                    "args": ["input", "output"],
                    "expected_result": {"exact_match": 0.0},
                    "comparer": "metric",
                },
            },
        ],
    },
]

@pytest.mark.asyncio
@pytest.mark.parametrize("monocle_test_case", agent_test_cases)
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY environment variable not set")
async def test_run_agents(monocle_test_case):
    await MonocleValidator().test_agent_async(root_agent, "google_adk", monocle_test_case)