import pytest
import requests
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError

from monocle_test_tools.evals.monocle import judge as judge_module
from monocle_test_tools.evals.monocle import template as template_loader
from monocle_test_tools.evals.monocle.judge import (
    TypeSafeJudge,
    LlmJudge,
    judge_for,
    response_model_for,
    run,
)
from monocle_test_tools.evals.monocle.judge import RunResult
from monocle_test_tools.evals.monocle.preprocess import Evidence
from monocle_test_tools.evals.monocle.template import EvalTemplate, EvidenceScope, LlmConfig, OutputField, TemplateError

LABELS = ["grounded", "partially_grounded", "ungrounded"]


def template_for(llm: LlmConfig, extra_fields: dict | None = None) -> EvalTemplate:
    structure_output = {
        "label": OutputField(description="verdict", type="string", enums=LABELS),
        "explanation": OutputField(description="why", type="string"),
    }
    structure_output.update(extra_fields or {})
    return EvalTemplate(
        name="grounding",
        eval_prompt="Judge the answer.",
        structure_output=structure_output,
        evidence_scope=EvidenceScope.default(),
        llm=llm,
    )


LLM_TEMPLATE = template_for(LlmConfig(model="openai:gpt-4.1"))
DECISION_TEMPLATE = template_for(LlmConfig(model="typesafe:jev-latest", min_confidence=0.8))
EVIDENCE = Evidence(trace_id="trace-1", text="the evidence")


def run_of(label: str, tokens: int = 0, explanation: str = "because") -> RunResult:
    return RunResult(label=label, explanation=explanation, judge_output={"label": label, "explanation": explanation},
                     total_tokens=tokens)


class FakeJudge:
    def __init__(self, runs: list[RunResult]):
        self.runs = list(runs)

    def judge(self, template, evidence) -> RunResult:
        return self.runs.pop(0)


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None, text: str = ""):
        self.status_code = status_code
        self.body = body
        self.text = text

    def json(self) -> dict:
        return self.body


def jev_body(choice: str, probabilities: dict[str, float]) -> dict:
    return {
        "answers": {"label": {"choice": choice, "probabilities": probabilities, "confidence": 0.9}},
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


class RecordingPost:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.calls = []

    def __call__(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.response


@pytest.fixture(autouse=True)
def typesafe_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")


def test_response_model_turns_enums_into_literal():
    model = response_model_for(LLM_TEMPLATE)
    assert model(label="grounded", explanation="x").label == "grounded"
    with pytest.raises(ValidationError):
        model(label="other", explanation="x")


def test_response_model_array_of_strings():
    template = template_for(
        LlmConfig(model="openai:gpt-4.1"),
        {"quotes": OutputField(description="quotes", type="array", items={"type": "string"})},
    )
    model = response_model_for(template)
    assert model(label="grounded", explanation="x", quotes=["a", "b"]).quotes == ["a", "b"]
    with pytest.raises(ValidationError):
        model(label="grounded", explanation="x", quotes="a")


def test_response_model_rejects_unknown_fields():
    model = response_model_for(LLM_TEMPLATE)
    with pytest.raises(ValidationError):
        model(label="grounded", explanation="x", surprise=1)


def test_decision_endpoint_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_URL", "https://jev.internal.example/")
    post = RecordingPost(FakeResponse(200, jev_body("grounded", {"grounded": 0.9})))
    TypeSafeJudge(DECISION_TEMPLATE.llm, post=post).judge(DECISION_TEMPLATE, EVIDENCE)
    assert post.calls[0]["url"] == "https://jev.internal.example/v1/systemone"


def test_decision_judge_builds_choice_question_from_enums():
    post = RecordingPost(FakeResponse(200, jev_body("grounded", {"grounded": 0.9})))
    TypeSafeJudge(DECISION_TEMPLATE.llm, post=post).judge(DECISION_TEMPLATE, EVIDENCE)
    call = post.calls[0]
    assert call["url"] == "https://api.typesafe.ai/v1/systemone"
    assert call["headers"]["Authorization"] == "Bearer secret"
    assert call["timeout"] == 30
    assert call["json"] == {
        "model": "jev-latest",
        "state": "the evidence",
        "questions": {"label": {
            "type": "choice",
            "instructions": "Judge the answer.",
            "criteria": {label: label for label in LABELS},
        }},
    }


def test_decision_judge_reads_choice_and_probabilities():
    probabilities = {"grounded": 0.1, "partially_grounded": 0.2, "ungrounded": 0.7}
    post = RecordingPost(FakeResponse(200, jev_body("ungrounded", probabilities)))
    run = TypeSafeJudge(DECISION_TEMPLATE.llm, post=post).judge(DECISION_TEMPLATE, EVIDENCE)
    assert run.label == "ungrounded"
    assert run.probabilities == probabilities
    assert run.total_tokens == 12
    assert run.explanation == "jev: grounded 0.10, partially_grounded 0.20, ungrounded 0.70"


def test_decision_judge_rejects_repeat():
    post = RecordingPost(FakeResponse(200, jev_body("grounded", {"grounded": 0.9})))
    with pytest.raises(ValueError, match="repeat"):
        run(TypeSafeJudge(DECISION_TEMPLATE.llm, post=post), DECISION_TEMPLATE, EVIDENCE, repeat=2)


def test_decision_judge_http_error_is_assertion_error():
    post = RecordingPost(FakeResponse(500, text="boom"))
    with pytest.raises(AssertionError, match="500.*boom"):
        TypeSafeJudge(DECISION_TEMPLATE.llm, post=post).judge(DECISION_TEMPLATE, EVIDENCE)


def failing_post(url, headers=None, json=None, timeout=None):
    raise requests.ConnectionError("connection refused")


def test_decision_judge_request_exception_is_assertion_error():
    with pytest.raises(AssertionError, match="decision model request failed: connection refused"):
        TypeSafeJudge(DECISION_TEMPLATE.llm, post=failing_post).judge(DECISION_TEMPLATE, EVIDENCE)


class NotJsonResponse(FakeResponse):
    def json(self) -> dict:
        raise requests.JSONDecodeError("Expecting value", "<html>", 0)


def test_decision_judge_non_json_answer_is_assertion_error():
    post = RecordingPost(NotJsonResponse(200, text="<html>gateway</html>"))
    with pytest.raises(AssertionError, match="not JSON.*Expecting value.*<html>gateway</html>"):
        TypeSafeJudge(DECISION_TEMPLATE.llm, post=post).judge(DECISION_TEMPLATE, EVIDENCE)


def test_decision_judge_missing_api_key_is_assertion_error(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY")
    with pytest.raises(AssertionError, match="TYPESAFE_API_KEY"):
        TypeSafeJudge(DECISION_TEMPLATE.llm, post=RecordingPost(FakeResponse(200))).judge(DECISION_TEMPLATE, EVIDENCE)


def test_decision_judge_unexpected_shape_is_assertion_error():
    post = RecordingPost(FakeResponse(200, {"answers": {}}))
    with pytest.raises(AssertionError, match="label"):
        TypeSafeJudge(DECISION_TEMPLATE.llm, post=post).judge(DECISION_TEMPLATE, EVIDENCE)


class ModelBuilder:
    def __init__(self, chat_model):
        self.chat_model = chat_model
        self.calls = []

    def __call__(self, model, **kwargs):
        self.calls.append({"model": model, **kwargs})
        return self.chat_model


def use_chat_model(monkeypatch, chat_model) -> ModelBuilder:
    builder = ModelBuilder(chat_model)
    monkeypatch.setattr(judge_module, "init_chat_model", builder)
    return builder


def test_judge_for_picks_judge_by_model(monkeypatch):
    use_chat_model(monkeypatch, FakeChatModel(None))
    assert isinstance(judge_for(DECISION_TEMPLATE.llm), TypeSafeJudge)
    assert isinstance(judge_for(LlmConfig(model="openai:gpt-4.1")), LlmJudge)


class FakeChatModel:
    def __init__(self, answer):
        self.answer = answer
        self.structured_kwargs = None
        self.messages = None

    def with_structured_output(self, response_model, **kwargs):
        self.response_model = response_model
        self.structured_kwargs = kwargs
        return self

    def invoke(self, messages):
        self.messages = messages
        return {"raw": FakeRawMessage(), "parsed": self.answer}


class FakeRawMessage:
    usage_metadata = {"total_tokens": 42}


class ToolCallingChatModel(BaseChatModel):
    """A chat model that keeps BaseChatModel.with_structured_output and answers with one tool call."""

    temperature: float | None = None
    max_tokens: int | None = None
    verdict: dict = {}

    @property
    def _llm_type(self) -> str:
        return "tool-calling-fake"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        tool_call = {"name": "grounding", "args": self.verdict, "id": "call-1"}
        message = AIMessage(
            content="",
            tool_calls=[tool_call],
            usage_metadata={"input_tokens": 30, "output_tokens": 12, "total_tokens": 42},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_llm_judge_sends_system_and_human_messages(monkeypatch):
    fake_model = FakeChatModel(response_model_for(LLM_TEMPLATE)(label="grounded", explanation="fine"))
    use_chat_model(monkeypatch, fake_model)
    run = LlmJudge(LLM_TEMPLATE.llm).judge(LLM_TEMPLATE, EVIDENCE)
    system_message, human_message = fake_model.messages
    assert (system_message.type, system_message.content) == ("system", "Judge the answer.")
    assert (human_message.type, human_message.content) == ("human", "the evidence")
    assert fake_model.structured_kwargs == {"include_raw": True}
    assert run.label == "grounded"
    assert run.explanation == "fine"
    assert run.judge_output == {"label": "grounded", "explanation": "fine"}
    assert run.total_tokens == 42


def test_llm_judge_builds_model_with_temperature_zero_and_given_params(monkeypatch):
    builder = use_chat_model(monkeypatch, FakeChatModel(None))
    LlmJudge(LlmConfig(model="openai:gpt-4.1"))
    LlmJudge(LlmConfig(model="openai:gpt-4.1", params={"temperature": 0.5, "top_p": 0.9}))
    assert builder.calls == [
        {"model": "openai:gpt-4.1", "temperature": 0},
        {"model": "openai:gpt-4.1", "temperature": 0.5, "top_p": 0.9},
    ]


def test_llm_judge_works_with_base_structured_output_and_params(monkeypatch):
    chat_model = ToolCallingChatModel(temperature=0, max_tokens=256, verdict={"label": "ungrounded", "explanation": "no"})
    builder = use_chat_model(monkeypatch, chat_model)
    template = template_for(LlmConfig(model="anthropic:claude-sonnet-4", params={"max_tokens": 256}))
    run = LlmJudge(template.llm).judge(template, EVIDENCE)
    assert builder.calls == [{"model": "anthropic:claude-sonnet-4", "temperature": 0, "max_tokens": 256}]
    assert run.label == "ungrounded"
    assert run.total_tokens == 42


def use_local_chat_model(monkeypatch, chat_model) -> ModelBuilder:
    from monocle_test_tools.harness import local_model

    builder = ModelBuilder(chat_model)
    monkeypatch.setattr(local_model, "local_chat_model", builder)
    return builder


def test_local_judge_is_served_on_this_machine_with_the_default_context(monkeypatch):
    hosted = use_chat_model(monkeypatch, FakeChatModel(None))
    local = use_local_chat_model(monkeypatch, FakeChatModel(None))

    LlmJudge(LlmConfig(model="local"))
    LlmJudge(LlmConfig(model="local:llama-3.2-3b", context_tokens=32768))

    assert local.calls == [
        {"model": "local", "context_tokens": 12288},
        {"model": "local:llama-3.2-3b", "context_tokens": 32768},
    ]
    assert hosted.calls == []


def test_llm_judge_builds_its_model_once_across_repeats(monkeypatch):
    builder = use_chat_model(monkeypatch, FakeChatModel(response_model_for(LLM_TEMPLATE)(label="grounded", explanation="ok")))
    runs = run(judge_for(LLM_TEMPLATE.llm), LLM_TEMPLATE, EVIDENCE, repeat=3)
    assert len(runs) == 3
    assert len(builder.calls) == 1


def test_llm_judge_unparseable_output_is_assertion_error(monkeypatch):
    fake_model = FakeChatModel(None)
    fake_model.invoke = lambda messages: {"raw": FakeRawMessage(), "parsed": None, "parsing_error": "bad label"}
    use_chat_model(monkeypatch, fake_model)
    with pytest.raises(AssertionError, match="did not match the template: bad label"):
        LlmJudge(LLM_TEMPLATE.llm).judge(LLM_TEMPLATE, EVIDENCE)


def test_decision_judge_answer_without_choice_is_assertion_error():
    post = RecordingPost(FakeResponse(200, {"answers": {"label": {"probabilities": {}}}}))
    with pytest.raises(AssertionError, match="no label choice"):
        TypeSafeJudge(DECISION_TEMPLATE.llm, post=post).judge(DECISION_TEMPLATE, EVIDENCE)


def test_decision_judge_choice_without_probability_is_assertion_error():
    post = RecordingPost(FakeResponse(200, jev_body("grounded", {"ungrounded": 1.0})))
    with pytest.raises(AssertionError, match="without a probability"):
        TypeSafeJudge(DECISION_TEMPLATE.llm, post=post).judge(DECISION_TEMPLATE, EVIDENCE)


def test_run_calls_the_judge_repeat_times_in_order():
    runs = run(FakeJudge([run_of("grounded"), run_of("ungrounded"), run_of("grounded")]), LLM_TEMPLATE, EVIDENCE, repeat=3)
    assert [each.label for each in runs] == ["grounded", "ungrounded", "grounded"]


def test_repeat_below_one_is_value_error():
    with pytest.raises(ValueError, match="at least 1"):
        run(FakeJudge([]), LLM_TEMPLATE, EVIDENCE, repeat=0)


def test_integer_param_stays_integer_from_template_to_model(monkeypatch):
    template = template_loader.load({
        "name": "grounding",
        "eval_prompt": "Judge the answer.",
        "structure_output": {
            "label": {"description": "verdict", "enums": LABELS},
            "explanation": {"description": "why"},
        },
        "llm": {"model": "openai:gpt-4.1-mini", "params": {"temperature": 0.2, "max_tokens": 2000}},
    })
    assert type(template.llm.params["max_tokens"]) is int
    assert type(template.llm.params["temperature"]) is float
    builder = use_chat_model(monkeypatch, FakeChatModel(None))

    LlmJudge(template.llm)

    built_max_tokens = builder.calls[0]["max_tokens"]
    assert built_max_tokens == 2000
    assert type(built_max_tokens) is int


def test_chat_model_refuses_min_confidence():
    with pytest.raises(TemplateError, match="min_confidence"):
        judge_for(LlmConfig(model="openai:gpt-4.1", min_confidence=0.8))


def test_decision_model_needs_min_confidence():
    with pytest.raises(TemplateError, match="min_confidence"):
        judge_for(LlmConfig(model="typesafe:judge"))


def test_decision_model_refuses_params():
    with pytest.raises(TemplateError, match="params"):
        judge_for(LlmConfig(model="typesafe:judge", min_confidence=0.8, params={"temperature": 0}))
