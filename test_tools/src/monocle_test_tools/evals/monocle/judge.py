"""Runs a judge over one trace's evidence: a chat model through LangChain, or a decision model through its provider's API.

A decision model answers once with a probability per label. Each provider is one class in DECISION_JUDGES,
keyed by the prefix of the template's llm.model; a chat model is anything LangChain's init_chat_model accepts."""
import os
from typing import Any, Literal

import requests
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, create_model

from monocle_test_tools.evals.monocle.preprocess import Evidence
from monocle_test_tools.evals.monocle.template import EvalTemplate, LlmConfig, OutputField, TemplateError


class RunResult(BaseModel):
    label: str
    explanation: str
    judge_output: dict
    probabilities: dict[str, float] = {}
    total_tokens: int = 0

DEFAULT_TEMPERATURE = 0
LOCAL_MODEL_PREFIX = "local"
LOCAL_MODEL_MESSAGE = "local judge models arrive with Monocle PR #831; use a provider:model string for now"

def run(judge, template: EvalTemplate, evidence: Evidence, repeat: int = 1) -> list[RunResult]:
    """Run the judge `repeat` times over the evidence; raise ValueError when a decision model is asked to repeat."""
    if repeat < 1:
        raise ValueError(f"repeat must be at least 1, got {repeat}")
    if getattr(judge, "answers_once", False) and repeat > 1:
        raise ValueError("repeat is not supported with a decision model: the same input gives the same answer")
    return [judge.judge(template, evidence) for _ in range(repeat)]


def judge_for(llm: LlmConfig):
    if is_decision_model(llm):
        return DECISION_JUDGES[llm.provider](llm)
    return LlmJudge(llm)


def is_decision_model(llm: LlmConfig) -> bool:
    return llm.provider in DECISION_JUDGES


class LlmJudge:
    answers_once = False

    def __init__(self, llm: LlmConfig):
        """Build the chat model once with temperature 0 unless llm.params says otherwise.

        Raise ValueError for a local model and TemplateError for a min_confidence, which only a decision model takes."""
        if llm.model.startswith(LOCAL_MODEL_PREFIX):
            raise ValueError(LOCAL_MODEL_MESSAGE)
        if llm.min_confidence is not None:
            raise TemplateError("'llm.min_confidence' only applies to a decision model.")
        self.llm = llm
        self.chat_model = init_chat_model(llm.model, **self._model_params())

    def judge(self, template: EvalTemplate, evidence: Evidence) -> RunResult:
        """Send the eval prompt and the evidence to the chat model; capture its structured verdict."""
        structured_model = self.chat_model.with_structured_output(response_model_for(template), include_raw=True)
        messages = [SystemMessage(template.eval_prompt), HumanMessage(evidence.text)]
        answer = structured_model.invoke(messages)
        if answer["parsed"] is None:
            raise AssertionError(f"'{template.name}' judge output did not match the template: {answer['parsing_error']}")
        verdict = answer["parsed"].model_dump()
        return RunResult(
            label=verdict["label"],
            explanation=verdict.get("explanation", ""),
            judge_output=verdict,
            total_tokens=(getattr(answer["raw"], "usage_metadata", None) or {}).get("total_tokens", 0),
        )

    def _model_params(self) -> dict[str, int | float]:
        return {"temperature": DEFAULT_TEMPERATURE, **self.llm.params}


class TypeSafeJudge:
    answers_once = True
    api_key_env = "TYPESAFE_API_KEY"
    api_url_env = "TYPESAFE_API_URL"
    default_api_url = "https://api.typesafe.ai"
    timeout_seconds = 30

    def __init__(self, llm: LlmConfig, post=requests.post):
        """Raise TemplateError when the llm block carries params, which a decision model does not take, or lacks min_confidence."""
        if llm.params:
            raise TemplateError("'llm.params' does not apply to a decision model; remove it.")
        if llm.min_confidence is None:
            raise TemplateError("A decision model needs 'llm.min_confidence'.")
        self.llm = llm
        self.post = post

    def judge(self, template: EvalTemplate, evidence: Evidence) -> RunResult:
        """Send the eval prompt, the labels and the evidence to TypeSafe's System One; capture its choice and probabilities."""
        response = self._post_request(template, evidence)
        if response.status_code != 200:
            raise AssertionError(f"decision model returned {response.status_code}: {response.text[:500]}")
        try:
            response_body = response.json()
        except ValueError as error:
            raise AssertionError(f"decision model answered with a body that is not JSON ({error}): {response.text[:500]}") from error
        return self._run_result_from(response_body)

    def _post_request(self, template: EvalTemplate, evidence: Evidence):
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise AssertionError(f"{self.api_key_env} is not set; the decision model needs it")
        base_url = os.environ.get(self.api_url_env, self.default_api_url).rstrip("/")
        try:
            return self.post(
                f"{base_url}/v1/systemone",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=self._request_body(template, evidence),
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as error:
            raise AssertionError(f"decision model request failed: {error}") from error

    def _request_body(self, template: EvalTemplate, evidence: Evidence) -> dict[str, Any]:
        return {
            "model": self.llm.model.partition(":")[2],
            "state": evidence.text,
            "questions": {
                "label": {
                    "type": "choice",
                    "instructions": template.eval_prompt,
                    "criteria": {label: label for label in template.labels},
                }
            },
        }

    def _run_result_from(self, response_body: dict) -> RunResult:
        answer = response_body.get("answers", {}).get("label")
        if answer is None or "choice" not in answer:
            raise AssertionError(f"decision model answer has no label choice: {response_body}")
        probabilities = answer.get("probabilities", {})
        if answer["choice"] not in probabilities:
            raise AssertionError(f"decision model chose '{answer['choice']}' without a probability for it: {response_body}")
        usage = response_body.get("usage", {})
        probability_text = ", ".join(f"{label} {probability:.2f}" for label, probability in probabilities.items())
        return RunResult(
            label=answer["choice"],
            explanation=f"jev: {probability_text}",
            judge_output=response_body,
            probabilities=probabilities,
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        )


def response_model_for(template: EvalTemplate) -> type[BaseModel]:
    """Build the pydantic model a judge must answer with; unknown fields are forbidden."""
    field_definitions = {name: _field_definition(field) for name, field in template.structure_output.items()}
    return create_model(template.name, __config__=ConfigDict(extra="forbid"), **field_definitions)


DECISION_JUDGES = {"typesafe": TypeSafeJudge}


def _field_definition(field: OutputField) -> tuple[Any, Any]:
    return _python_type_of(field), Field(description=field.description)


def _python_type_of(field: OutputField) -> Any:
    scalar_types = {"number": float, "integer": int, "boolean": bool}
    if field.enums:
        enum_type = Literal[tuple(field.enums)]
        return list[enum_type] if field.type == "array" else enum_type
    if field.type == "array":
        return list[scalar_types.get((field.items or {}).get("type"), str)]
    return scalar_types.get(field.type, str)
