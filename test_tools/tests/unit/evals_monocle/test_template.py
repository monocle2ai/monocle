import copy
import json
import logging
from pathlib import Path

import pytest

from monocle_test_tools.evals.monocle.template import EvalTemplate, EvidenceScope, TemplateError, coherence_warnings, load

EXAMPLE_TEMPLATE = Path(__file__).parents[2] / "integration" / "custom_templates" / "refund_confirmation_grounding.json"


def minimal_template() -> dict:
    return {
        "name": "tone_check",
        "eval_prompt": "Judge the tone of the reply.",
        "structure_output": {
            "label": {"description": "tone", "enums": ["polite", "rude"]},
            "explanation": {"description": "why"},
        },
        "llm": {"model": "openai:gpt-4.1-mini"},
    }


def load_with(**changes) -> EvalTemplate:
    template = copy.deepcopy(minimal_template())
    template.update(changes)
    return load(template)


def rejected_message(**changes) -> str:
    with pytest.raises(TemplateError) as rejection:
        load_with(**changes)
    return str(rejection.value)


def test_minimal_valid_template_loads():
    template = load(minimal_template())
    assert template.name == "tone_check"
    assert template.evidence_scope == EvidenceScope.default()
    assert template.llm.model == "openai:gpt-4.1-mini"


def test_example_template_file_loads():
    template = load(EXAMPLE_TEMPLATE)
    assert template.name == "refund_confirmation_grounding"
    assert template.evidence_scope.roles == ["user", "assistant", "tool"]


def test_template_without_llm_model_names_the_missing_field():
    template = minimal_template()
    del template["llm"]
    with pytest.raises(TemplateError, match="llm.model"):
        load(template)


def test_name_with_double_underscore_rejected():
    assert "name" in rejected_message(name="tone__check")


def test_name_over_50_chars_rejected():
    assert "name" in rejected_message(name="a" * 51)


def test_empty_eval_prompt_rejected():
    assert "eval_prompt" in rejected_message(eval_prompt="   ")


def test_structure_output_without_label_rejected():
    assert "label" in rejected_message(structure_output={"explanation": {"description": "why"}})


def test_label_without_enums_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone"},
        "explanation": {"description": "why"},
    })
    assert "enums" in message


def test_label_enum_not_identifier_safe_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone", "enums": ["no hallucination", "hallucination"]},
        "explanation": {"description": "why"},
    })
    assert "no hallucination" in message


def test_duplicate_label_enums_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone", "enums": ["polite", "polite"]},
        "explanation": {"description": "why"},
    })
    assert "duplicate" in message


def test_structure_output_without_explanation_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone", "enums": ["polite", "rude"]},
    })
    assert "explanation" in message


def test_field_without_description_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone", "enums": ["polite", "rude"]},
        "explanation": {},
    })
    assert "description" in message


def test_field_with_unknown_type_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone", "enums": ["polite", "rude"]},
        "explanation": {"description": "why", "type": "tuple"},
    })
    assert "type" in message


def test_unknown_top_level_key_rejected():
    assert "colour" in rejected_message(colour="red")


def test_evidence_scope_missing_roles_rejected():
    message = rejected_message(evidence_scope={"span_types": ["inference"], "events": ["input"]})
    assert "roles" in message


def test_evidence_scope_unknown_span_type_rejected():
    message = rejected_message(evidence_scope={
        "span_types": ["made.up"], "roles": ["user"], "events": ["input"],
    })
    assert "made.up" in message


def test_evidence_scope_group_by_rejected():
    message = rejected_message(evidence_scope={
        "span_types": ["inference"], "roles": ["user"], "events": ["input"], "group_by": "traces",
    })
    assert "group_by" in message


def test_llm_params_outside_allow_list_rejected():
    message = rejected_message(llm={"model": "openai:gpt-4.1", "params": {"frequency_penalty": 1}})
    assert "frequency_penalty" in message


def test_llm_params_non_number_rejected():
    message = rejected_message(llm={"model": "openai:gpt-4.1", "params": {"temperature": "hot"}})
    assert "temperature" in message


def test_llm_params_boolean_rejected():
    message = rejected_message(llm={"model": "openai:gpt-4.1", "params": {"max_tokens": True}})
    assert "'llm.params.max_tokens' must be a number." in message


@pytest.mark.parametrize("min_confidence", [1.5, 0])
def test_min_confidence_out_of_range_rejected(min_confidence):
    message = rejected_message(llm={"model": "typesafe:judge", "min_confidence": min_confidence})
    assert "min_confidence" in message


def test_outer_template_key_is_unwrapped(tmp_path):
    template_file = tmp_path / "wrapped.json"
    template_file.write_text(json.dumps({"template": minimal_template()}))
    assert load(template_file).name == "tone_check"


def test_dict_template_loads():
    assert load(minimal_template()).labels == ["polite", "rude"]


def test_labels_property_returns_enums():
    template = load_with(structure_output={
        "label": {"description": "grounding", "enums": ["grounded", "ungrounded"]},
        "explanation": {"description": "why"},
    })
    assert template.labels == ["grounded", "ungrounded"]


def test_coherence_warns_when_prompt_mentions_tools_but_scope_excludes_them():
    template = load_with(
        eval_prompt="Check every tool call the agent made.",
        evidence_scope={"span_types": ["inference"], "roles": ["user", "assistant"], "events": ["input", "output"]},
    )
    warnings = coherence_warnings(template)
    assert len(warnings) == 1
    assert "agentic.tool.invocation" in warnings[0]


def test_load_logs_each_coherence_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="monocle_test_tools.evals.monocle.template"):
        load_with(
            eval_prompt="Check every tool call the agent made.",
            evidence_scope={"span_types": ["inference"], "roles": ["user", "assistant"], "events": ["input", "output"]},
        )
    warning_records = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warning_records) == 1
    assert "tone_check" in warning_records[0].getMessage()
    assert "agentic.tool.invocation" in warning_records[0].getMessage()


def test_load_logs_nothing_for_a_coherent_template(caplog):
    with caplog.at_level(logging.WARNING, logger="monocle_test_tools.evals.monocle.template"):
        load(minimal_template())
    assert caplog.records == []


def test_coherence_is_silent_for_default_scope():
    template = load_with(eval_prompt="Check every tool call, the system prompt, MCP and retrieved documents.")
    assert coherence_warnings(template) == []


def scoped_template(prompt: str, span_types: list[str], roles: list[str], events: list[str]) -> EvalTemplate:
    return load_with(
        eval_prompt=prompt,
        evidence_scope={"span_types": span_types, "roles": roles, "events": events},
    )


def test_coherence_warns_when_prompt_mentions_mcp_but_scope_excludes_it():
    template = scoped_template("Check the MCP calls.", ["inference"], ["user"], ["input"])
    assert "agentic.mcp.invocation" in coherence_warnings(template)[0]


def test_coherence_warns_when_prompt_mentions_retrieval_but_scope_excludes_it():
    template = scoped_template("Check the retrieved context.", ["inference"], ["user"], ["input"])
    assert "retrieval" in coherence_warnings(template)[0]


def test_coherence_warns_when_prompt_mentions_user_question_but_scope_excludes_input():
    template = scoped_template("Compare against the user question.", ["inference"], ["user"], ["output"])
    assert "user's question" in coherence_warnings(template)[0]


def test_coherence_warns_when_prompt_mentions_system_prompt_but_scope_excludes_it():
    template = scoped_template("Follow the system prompt.", ["inference"], ["user"], ["input"])
    assert "system" in coherence_warnings(template)[0]


def test_coherence_tool_rule_is_quiet_when_only_the_tool_span_type_is_present():
    template = scoped_template("Check every tool call.", ["agentic.tool.invocation"], ["user"], ["input"])
    assert coherence_warnings(template) == []


def test_coherence_tool_rule_is_quiet_when_only_the_tool_role_is_present():
    template = scoped_template("Check every tool call.", ["inference"], ["tool"], ["input"])
    assert coherence_warnings(template) == []


def test_upper_case_name_rejected():
    assert "name" in rejected_message(name="Tone_Check")


def test_label_declared_with_enum_loads():
    template = load_with(structure_output={
        "label": {"description": "tone", "enum": ["polite", "rude"]},
        "explanation": {"description": "why"},
    })
    assert template.labels == ["polite", "rude"]


def test_field_with_both_enum_and_enums_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone", "enum": ["polite"], "enums": ["rude"]},
        "explanation": {"description": "why"},
    })
    assert "enums" in message


def test_items_that_is_not_an_object_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone", "enums": ["polite", "rude"]},
        "explanation": {"description": "why", "items": "string"},
    })
    assert "items" in message


def test_any_of_that_is_not_a_list_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone", "enums": ["polite", "rude"]},
        "explanation": {"description": "why", "anyOf": "string"},
    })
    assert "anyOf" in message


def test_unhashable_field_type_rejected():
    message = rejected_message(structure_output={
        "label": {"description": "tone", "enums": ["polite", "rude"]},
        "explanation": {"description": "why", "type": ["string"]},
    })
    assert "type" in message


@pytest.mark.parametrize("min_confidence", ["high", True])
def test_min_confidence_that_is_not_a_number_rejected(min_confidence):
    message = rejected_message(llm={"model": "typesafe:judge", "min_confidence": min_confidence})
    assert "min_confidence" in message
