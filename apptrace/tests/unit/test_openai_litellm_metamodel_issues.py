"""
Unit tests reproducing data-type handling issues in the OpenAI and LiteLLM
metamodels (instrumentation/metamodel/openai and instrumentation/metamodel/litellm).

Each test documents a bug found in a code review of entity-attribute accessors
and event accessors. Tests are written to FAIL against the current
implementation and PASS once the underlying bug is fixed.

Run only this file:
    pytest apptrace/tests/unit/test_openai_litellm_metamodel_issues.py -v
"""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from monocle_apptrace.instrumentation.metamodel.openai import (
    _helper as openai_helper,
)
from monocle_apptrace.instrumentation.metamodel.openai.entities.inference import (
    INFERENCE as OPENAI_INFERENCE,
)
from monocle_apptrace.instrumentation.metamodel.openai.entities.retrieval import (
    RETRIEVAL as OPENAI_RETRIEVAL,
)
from monocle_apptrace.instrumentation.metamodel.litellm import (
    _helper as litellm_helper,
)
from monocle_apptrace.instrumentation.metamodel.litellm.entities.inference import (
    INFERENCE as LITELLM_INFERENCE,
)


# ---------------------------------------------------------------------------
# Helpers for building accessor `arguments` payloads + flat lookups of the
# nested entity dicts (attribute lists / event lists).
# ---------------------------------------------------------------------------

def _args(instance=None, kwargs=None, result=None, exception=None, args=()):
    return {
        "instance": instance if instance is not None else SimpleNamespace(),
        "kwargs": kwargs if kwargs is not None else {},
        "args": args,
        "result": result,
        "exception": exception,
    }


def _find_attr(entity, attribute_name, group_comment=None):
    """Find the first attribute dict matching attribute_name (optionally
    restricted to a group identified by its `_comment` marker)."""
    for group in entity["attributes"]:
        if group_comment and not any(
            isinstance(a, dict) and a.get("_comment") == group_comment for a in group
        ):
            continue
        for attr in group:
            if attr.get("attribute") == attribute_name:
                return attr
    raise AssertionError(
        f"attribute={attribute_name!r} (group_comment={group_comment!r}) not found"
    )


def _find_event_attr(entity, event_name, attribute_name=None):
    for event in entity["events"]:
        if event["name"] == event_name:
            if attribute_name is None:
                return event["attributes"][0]
            for attr in event["attributes"]:
                if attr.get("attribute") == attribute_name:
                    return attr
    raise AssertionError(f"event={event_name!r} attribute={attribute_name!r} not found")


def _build_openai_instance(base_url="https://api.openai.com", api_version=None):
    """Construct a minimal stand-in for an OpenAI client instance."""
    client = MagicMock()
    client.base_url = base_url
    if api_version is not None:
        client._api_version = api_version
    else:
        # ensure it does NOT have _api_version to avoid azure detection
        del client._api_version
    instance = MagicMock()
    instance._client = client
    instance.base_url = base_url
    return instance


# ===========================================================================
# LiteLLM metamodel bugs
# ===========================================================================

class TestLiteLLMMetamodelBugs:
    """Bugs in apptrace/src/monocle_apptrace/instrumentation/metamodel/litellm."""

    def test_deployment_accessor_does_not_crash_on_dict_kwargs(self):
        """Bug: litellm/entities/inference.py:33 calls
            resolve_from_alias(arguments["kwargs"].__dict__, [...])
        but `arguments["kwargs"]` is a plain dict, and `dict.__dict__` raises
        AttributeError. The accessor should read from the dict directly.
        """
        deployment_attr = _find_attr(LITELLM_INFERENCE, "deployment")
        accessor = deployment_attr["accessor"]
        args = _args(kwargs={"engine": "my-deployment"})
        # Today this raises AttributeError: 'dict' object has no attribute '__dict__'
        result = accessor(args)
        assert result == "my-deployment", (
            "deployment accessor should resolve from kwargs; currently it does "
            "`kwargs.__dict__` which raises AttributeError on a plain dict."
        )

    def test_update_span_from_llm_response_total_tokens_missing(self):
        """Bug: litellm/_helper.py:92 uses `getattr(token_usage, 'total_tokens')`
        with NO default. If a provider omits total_tokens (e.g. only exposes
        input/output tokens), this raises AttributeError and the entire
        metadata event is lost. Compare with lines 90-91 which pass `None`.
        """
        usage = SimpleNamespace(completion_tokens=10, prompt_tokens=20)
        # deliberately no total_tokens attribute
        response = SimpleNamespace(usage=usage)
        # Today this raises AttributeError
        meta = litellm_helper.update_span_from_llm_response(response)
        assert meta.get("completion_tokens") == 10
        assert meta.get("prompt_tokens") == 20
        assert meta.get("total_tokens") is None, (
            "Missing total_tokens should yield None, not propagate AttributeError."
        )

    def test_model_llm_type_accessor_handles_missing_model(self):
        """Bug: litellm/entities/inference.py:62 builds
            'model.llm.' + resolve_from_alias(kwargs, [...])
        If none of the aliases are present, resolve_from_alias returns None
        and the concatenation raises TypeError. The accessor must produce a
        string regardless.
        """
        model_type_attr = _find_attr(
            LITELLM_INFERENCE, "type", group_comment="LLM Model"
        )
        accessor = model_type_attr["accessor"]
        # No 'model'/'model_name'/'endpoint_name'/'deployment_name' present.
        args = _args(kwargs={"temperature": 0.7})
        # Today this raises TypeError: can only concatenate str (not "NoneType") to str
        result = accessor(args)
        assert isinstance(result, str), (
            "Model-type accessor must return a string even when model alias is missing."
        )

    def test_extract_assistant_message_content_none_with_tool_calls(self):
        """Bug: litellm/_helper.py:50 stores `message.content` directly. When
        the assistant emits ONLY tool_calls, content is None, producing
        '{"assistant": null}' as the span's data.output.response, which is
        valid JSON but loses the tool_calls payload and surprises consumers
        that expect a string body or the tool payload.
        """
        message = SimpleNamespace(
            role="assistant",
            content=None,  # tool-call response shape
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(name="book_flight", arguments='{"to":"LAX"}')
                )
            ],
        )
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        response = SimpleNamespace(choices=[choice])
        args = _args(result=response)

        out = litellm_helper.extract_assistant_message(args)
        assert isinstance(out, str), "Assistant message accessor must return a string."
        assert "null" not in out, (
            "Response with tool_calls should not serialize as {\"assistant\": null}; "
            "the tool payload (or an empty string) should be emitted instead."
        )

    def test_extract_assistant_message_error_attr_returns_string(self):
        """Bug: litellm/_helper.py:55-56 returns `arguments['result'].error`
        directly. The SDK error object may be a dict / Pydantic model /
        Exception; OpenTelemetry span attributes only accept str/bool/int/
        float/sequence-thereof. The accessor must coerce to a string.
        """
        error_obj = {"code": "rate_limited", "message": "Too many requests"}
        # status != success → error branch; no exception → fall through to .error
        bad_result = SimpleNamespace(status="error", error=error_obj)
        args = _args(result=bad_result, exception=None)

        out = litellm_helper.extract_assistant_message(args)
        assert isinstance(out, str), (
            "Error-path return must be a string; today it can be a dict/object "
            "which OpenTelemetry will reject as a span attribute value."
        )

    def test_extract_finish_reason_returns_consistent_type_on_exception(self):
        """Divergence with OpenAI: litellm/_helper.py:107-108 returns the
        literal string 'error' when exception is set, while openai/_helper.py
        returns None on the same condition. The two metamodels disagree on the
        finish_reason value space, and downstream consumers must special-case.
        The two should converge (both None, or both 'error').
        """
        args = _args(exception=RuntimeError("boom"))
        litellm_value = litellm_helper.extract_finish_reason(args)
        openai_value = openai_helper.extract_finish_reason(args)
        assert litellm_value == openai_value, (
            f"OpenAI and LiteLLM disagree on finish_reason for exception path: "
            f"openai={openai_value!r}, litellm={litellm_value!r}. Pick one."
        )


# ===========================================================================
# OpenAI metamodel bugs
# ===========================================================================

class TestOpenAIMetamodelBugs:
    """Bugs in apptrace/src/monocle_apptrace/instrumentation/metamodel/openai."""

    def test_agent_inference_type_handles_empty_assistant_message(self):
        """Bug: openai/_helper.py:313 calls
            json.loads(extract_assistant_message(arguments))
        with NO try/except. extract_assistant_message returns None on caught
        exceptions (line 190) and "" when no messages are collected
        (line 179). Both inputs crash json.loads (TypeError / JSONDecodeError).
        Since agent_inference_type is used as the span SUBTYPE (entities/
        inference.py:44), every inference that fails to extract a message
        fails to build a span at all.
        """
        # Force the "no messages collected" path: success status, response with
        # no tools/output/output_text/choices attributes.
        result = SimpleNamespace(status="success")  # no choices/output/etc.
        args = _args(result=result, exception=None)
        # Today raises json.decoder.JSONDecodeError: Expecting value: line 1 column 1
        subtype = openai_helper.agent_inference_type(args)
        assert subtype in {"turn_end", "tool_call", "delegation"}, (
            f"agent_inference_type must return a known subtype, got {subtype!r}; "
            "today it crashes on json.loads of '' / None."
        )

    def test_inference_type_accessor_handles_none_inference_type(self):
        """Bug: openai/entities/inference.py:52-55
            'inference.' + (_helper.get_inference_type(...)) or 'openai'
        Operator precedence: the `or 'openai'` is on the whole `str + X`
        expression, not on the inner X. If get_inference_type returns None,
        `'inference.' + None` raises TypeError and the `or 'openai'` fallback
        is never reached.
        """
        type_attr = _find_attr(
            OPENAI_INFERENCE,
            "type",
            group_comment="provider type ,name , deployment , inference_endpoint",
        )
        accessor = type_attr["accessor"]

        # Build an instance whose _client lacks _api_version and base_url so
        # get_inference_type may legitimately return a falsy value. We patch
        # the helper to deterministically return None to reproduce.
        instance = _build_openai_instance(base_url=None)
        args = _args(instance=instance)

        original = openai_helper.get_inference_type
        openai_helper.get_inference_type = lambda _inst: None
        try:
            # Today raises TypeError: can only concatenate str (not "NoneType") to str
            value = accessor(args)
        finally:
            openai_helper.get_inference_type = original

        assert isinstance(value, str), "inference type accessor must return a string."
        assert value == "inference.openai" or value.startswith("inference."), (
            "Fallback should produce a valid 'inference.<provider>' value."
        )

    def test_model_llm_type_accessor_handles_missing_model(self):
        """Bug: openai/entities/inference.py:94-100 builds
            'model.llm.' + resolve_from_alias(kwargs, [...])
        If model alias is absent, the concatenation raises TypeError.
        """
        model_type_attr = _find_attr(
            OPENAI_INFERENCE, "type", group_comment="LLM Model"
        )
        accessor = model_type_attr["accessor"]
        args = _args(kwargs={"temperature": 0.0})  # no model alias
        # Today raises TypeError
        result = accessor(args)
        assert isinstance(result, str)

    def test_update_span_from_llm_response_usage_none_no_response_metadata(self):
        """Bug: openai/_helper.py:243-249. When `response.usage` is present
        but None, the code falls through to `response.response_metadata`
        WITHOUT a hasattr guard. Real OpenAI SDK responses do not carry
        `response_metadata`, so this raises AttributeError. The function has
        no try/except.
        """
        # Use SimpleNamespace because MagicMock would auto-create attributes.
        response = SimpleNamespace(usage=None)
        # Today raises AttributeError: 'types.SimpleNamespace' object has no
        # attribute 'response_metadata'
        meta = openai_helper.update_span_from_llm_response(response)
        assert isinstance(meta, dict), (
            "update_span_from_llm_response must return a dict (possibly empty) "
            "when usage is missing; today it raises AttributeError."
        )

    def test_extract_assistant_message_output_text_none(self):
        """Bug: openai/_helper.py:164
            if hasattr(response, 'output_text') and len(response.output_text):
        `hasattr` only confirms presence; the value can still be None.
        `len(None)` raises TypeError which is NOT in the caught exception
        tuple `(IndexError, AttributeError)`, so it propagates out of the
        function and aborts span attribute extraction.
        """
        # Build a result where output_text exists but is None.
        response = SimpleNamespace(status="success", output_text=None)
        args = _args(result=response, exception=None)
        # Today raises TypeError: object of type 'NoneType' has no len()
        out = openai_helper.extract_assistant_message(args)
        # After fix: should not crash; empty/none-content can yield "" or None
        # but must not propagate TypeError.
        assert out is None or isinstance(out, str)

    def test_extract_assistant_message_error_attr_returns_string(self):
        """Bug: openai/_helper.py:183-184 returns `arguments['result'].error`
        directly. May be a non-string object (dict, Pydantic model). Must be
        coerced to a string for span attribute use.
        """
        error_obj = {"code": "rate_limited"}
        bad_result = SimpleNamespace(status="error", error=error_obj)
        args = _args(result=bad_result, exception=None)
        out = openai_helper.extract_assistant_message(args)
        assert isinstance(out, str), (
            "Error path must coerce result.error to a string for OTel compatibility."
        )

    def test_extract_assistant_message_never_returns_none(self):
        """Bug: openai/_helper.py:135-190 has a return path that yields None
        (line 190, on caught IndexError/AttributeError). Span event accessors
        should not return None — OpenTelemetry rejects None attribute values
        and the downstream agent_inference_type does json.loads on the result
        and crashes on None.
        """
        # Force the AttributeError path: `response.output` is a non-empty list
        # whose items don't carry a `.type` attribute. The function does
        # `response_message.type == "function_call"` unguarded, triggering
        # AttributeError → caught → returns None.
        response = SimpleNamespace(
            status="success",
            output=[SimpleNamespace()],  # no `.type` attribute
        )
        args = _args(result=response, exception=None)
        out = openai_helper.extract_assistant_message(args)
        assert out is not None, (
            "extract_assistant_message must not return None; consumers "
            "(agent_inference_type → json.loads) crash on None."
        )

    def test_extract_assistant_message_with_choices_content_none(self):
        """Bug: openai/_helper.py:178 stores `choices[0].message.content` as
        the value of `{role: ...}`. When the assistant only emits tool_calls,
        `content` is None. The serialized event becomes '{"assistant": null}',
        losing the tool_calls and yielding a null-typed span value.
        """
        message = SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(name="search", arguments='{"q":"x"}')
                )
            ],
        )
        choice = SimpleNamespace(message=message, finish_reason="tool_calls")
        response = SimpleNamespace(status="success", choices=[choice])
        args = _args(result=response, exception=None)

        out = openai_helper.extract_assistant_message(args)
        assert isinstance(out, str)
        assert "null" not in out, (
            "When content is None but tool_calls are present, the accessor "
            "should emit the tool payload (or empty string), not '{\"assistant\": null}'."
        )

    def test_extract_assistant_message_collects_all_messages(self):
        """Bug: openai/_helper.py:179
            return get_json_dumps(messages[0]) if messages else ""
        The function appends multiple messages (tools, output, output_text,
        choices) but returns only the FIRST. For a Responses-API response
        carrying both a function_call (output) and a textual answer
        (output_text), the textual answer is dropped.
        """
        function_call_item = SimpleNamespace(
            type="function_call",
            call_id="call_1",
            name="lookup",
            arguments='{"q":"x"}',
        )
        response = SimpleNamespace(
            status="success",
            output=[function_call_item],
            output_text="here is the textual answer",
            role="assistant",
        )
        args = _args(result=response, exception=None)
        out = openai_helper.extract_assistant_message(args)
        assert isinstance(out, str)
        assert "here is the textual answer" in out, (
            "When both tool-call output AND output_text are present, the "
            "accessor must include both; today only the first appended "
            "message is returned (the tool list), dropping the assistant text."
        )

    def test_extract_messages_tool_call_arguments_not_double_encoded(self):
        """Bug: openai/_helper.py:121-124. `tool_call.function.arguments` is
        ALREADY a JSON string per the OpenAI SDK. Wrapping the parent dict in
        get_json_dumps produces doubly-escaped output like
            '{"tool_arguments": "{\\"city\\":\\"Mumbai\\"}"}'
        Compare with the function_call branch at line 102-107 which stores
        arguments directly. Choose one canonical form.
        """
        tool_call = SimpleNamespace(
            function=SimpleNamespace(
                name="book_flight",
                arguments='{"city":"Mumbai"}',
            )
        )
        kwargs = {
            "messages": [
                {"role": "assistant", "tool_calls": [tool_call]},
            ]
        }
        msgs = openai_helper.extract_messages(kwargs)
        # Decode the assistant message; the tool_arguments string should be a
        # JSON object representation, not a doubly-escaped string.
        assert msgs, "extract_messages should produce at least one message"
        decoded = json.loads(msgs[0])
        assistant_payload = decoded.get("assistant")
        # The current implementation wraps each tool call as a JSON STRING
        # inside the list. The fix should make it a JSON OBJECT (or any
        # canonical, non-double-encoded representation).
        assert assistant_payload, "Assistant message missing in payload"
        first = assistant_payload[0]
        assert not isinstance(first, str), (
            "Tool-call entries are doubly JSON-encoded; the inner element "
            "should be a dict, not a JSON-encoded string."
        )

    def test_update_input_span_events_handles_non_string_list(self):
        """Bug: openai/_helper.py:225-228
            query = ' '.join(kwargs['input'])
        Assumes every element is a string. For OpenAI embedding batch calls
        the input is often a list of token-id arrays (list[list[int]]) or a
        list of dicts. `str.join` raises TypeError on non-string elements,
        and there's no try/except. Used by the retrieval span data.input.
        """
        retrieval_input_attr = _find_event_attr(
            OPENAI_RETRIEVAL, "data.input", "input"
        )
        accessor = retrieval_input_attr["accessor"]

        # Case 1: list of dicts
        args1 = _args(kwargs={"input": [{"a": 1}, {"b": 2}]})
        # Today raises TypeError: sequence item 0: expected str instance, dict found
        out1 = accessor(args1)
        assert out1 is None or isinstance(out1, str)

        # Case 2: list of token-id arrays (embedding batch shape)
        args2 = _args(kwargs={"input": [[1, 2, 3], [4, 5, 6]]})
        out2 = accessor(args2)
        assert out2 is None or isinstance(out2, str)

    def test_extract_inference_endpoint_handles_missing_client_attr(self):
        """Bug: openai/_helper.py:209-214
            if inference_endpoint.is_none() and "meta" in instance.client.__dict__:
        The `instance.client.__dict__` access is unguarded — if `instance`
        has no `client` attribute (or `client` uses __slots__), this raises
        AttributeError. Only the first half (instance._client) is wrapped in
        try_option.
        """
        # _client exists but base_url resolves to None, forcing the broken
        # right-hand side of the AND. `client` attribute is absent.
        client = SimpleNamespace(base_url=None)
        instance = SimpleNamespace(_client=client)  # no `client` attribute
        # Today raises AttributeError: 'types.SimpleNamespace' object has no
        # attribute 'client'
        endpoint = openai_helper.extract_inference_endpoint(instance)
        # After fix: should fall through to provider_name fallback without crashing.
        assert endpoint is None or isinstance(endpoint, str)


# ===========================================================================
# Cross-cutting / shared concerns
# ===========================================================================

class TestSpanAttributeContractCompatibility:
    """Span attribute values must be str/bool/int/float or sequences thereof
    (OpenTelemetry contract). These tests verify that accessors do not return
    forbidden None / object values along common code paths.
    """

    def test_openai_provider_name_returns_string_fallback(self):
        """Bug: openai/_helper.py:193-206 extract_provider_name returns None
        when no base_url/host is resolvable. The entity accessor
        (entities/inference.py:57-60) pipes this value straight into a span
        attribute. OpenTelemetry rejects None — span attributes must be
        str/bool/int/float/sequence. The function should return a string
        fallback (e.g., "unknown") or the accessor should coerce.
        """
        instance = SimpleNamespace(_client=SimpleNamespace(base_url=None))
        result = openai_helper.extract_provider_name(instance)
        assert isinstance(result, str), (
            "Provider-name accessor must return a string (today returns None, "
            "which OpenTelemetry rejects as a span attribute value)."
        )

    def test_litellm_extract_provider_name_handles_non_string(self):
        """litellm/_helper.py:64-68 does `url.split('//')`. If `url` is a
        non-string (e.g., a URL object that just happens to have a host
        attribute), this raises AttributeError. Test that the function is
        robust to None and to non-strings.
        """
        # None is guarded today
        assert litellm_helper.extract_provider_name(None) is None
        # Non-string input is NOT guarded
        url_obj = SimpleNamespace(host="api.example.com")
        # Today: AttributeError: 'types.SimpleNamespace' object has no attribute 'split'
        result = litellm_helper.extract_provider_name(url_obj)
        assert result is None or isinstance(result, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
