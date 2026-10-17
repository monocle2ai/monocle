""""local" runs an open model on this machine, opt-in; each model's tool calls are understood; unit tests
never start a real model."""
import sys

import pytest
from langchain_core.messages import AIMessage

from monocle_test_tools.harness import local_model
from monocle_test_tools.harness.driver import resolve_model


def test_nothing_configured_is_an_error_that_names_local(monkeypatch):
    monkeypatch.delenv("MONOCLE_TEST_AGENT_MODEL", raising=False)
    with pytest.raises(ValueError, match="model=.*MONOCLE_TEST_AGENT_MODEL.*'local'"):
        resolve_model(None)


def test_local_by_argument_or_env_var_starts_the_local_model(monkeypatch):
    started = []
    monkeypatch.setattr(local_model, "local_chat_model", lambda spec: started.append(spec) or "LOCAL")
    monkeypatch.setenv("MONOCLE_TEST_AGENT_MODEL", "local:llama-3.2-3b")
    assert resolve_model("local") == "LOCAL" and resolve_model(None) == "LOCAL"
    assert started == ["local", "local:llama-3.2-3b"]


def test_an_unknown_local_model_names_the_known_ones():
    with pytest.raises(ValueError, match="known: local, local:qwen3-4b"):
        local_model.local_chat_model("local:nope")


def test_the_install_is_a_pinned_prebuilt_wheel_into_this_interpreter():
    command = local_model._install_command()
    assert command[0] in (sys.executable, "uv")
    assert f"llama-cpp-python[server]=={local_model.RUNTIME_VERSION}" in command
    assert command[command.index("--only-binary") + 1] == "llama-cpp-python"


def test_a_second_local_model_in_one_run_is_refused(monkeypatch):
    monkeypatch.setattr(local_model, "_server", type("Running", (), {"poll": lambda self: None})())
    monkeypatch.setattr(local_model, "_port", 1234)
    monkeypatch.setattr(local_model, "_model_path", "/cache/qwen.gguf")
    assert local_model.ensure_server("/cache/qwen.gguf") == 1234
    with pytest.raises(ValueError, match="already running"):
        local_model.ensure_server("/cache/phi.gguf")


@pytest.mark.parametrize("text", [
    '<tool_call>\n{"name": "date", "arguments": {}}\n</tool_call>',        # Qwen
    '[{"type": "function", "name": "date", "arguments": {}}]',             # Phi
    '<|python_tag|>{"name": "date", "parameters": {}}',                    # Llama
])
def test_each_models_tool_call_format_becomes_a_tool_call(text):
    message = AIMessage(content=text)
    local_model.lift_tool_calls(message, {"date"})
    assert [c["name"] for c in message.tool_calls] == ["date"] and message.content == ""


def test_plain_text_and_json_naming_no_offered_tool_stay_text():
    for text in ("Mumbai, please.", 'My booking: {"name": "Sam", "parameters": {"seat": "aisle"}}'):
        message = AIMessage(content=text)
        local_model.lift_tool_calls(message, {"date"})
        assert message.tool_calls == [] and message.content == text
