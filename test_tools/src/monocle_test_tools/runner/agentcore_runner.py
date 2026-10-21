import asyncio
import json
import logging
import os
from typing import Any, Optional, Union

from monocle_apptrace.instrumentation.common.constants import (
    AGENTCORE_CUSTOM_HEADER_PREFIX,
    MONOCLE_TRACE_RETRIEVAL_KEY_ENV,
    TRACE_RETURN_REQUEST_HEADER,
)
from monocle_test_tools.file_span_loader import JSONSpanLoader
from monocle_test_tools.runner.agent_runner import AgentRunner

logger = logging.getLogger(__name__)

# A deployed agent running with trace-return enabled appends its spans after a
# delimiter of the form ``__MONOCLE_TRACES__<hex>__``. boto3 surfaces no response
# header to announce that delimiter, so the fixed prefix is what locates it.
_DELIMITER_PREFIX = "__MONOCLE_TRACES__"
_DELIMITER_SUFFIX = "__"

# Okahu indexes the deployed agent's session scope under this fact name.
REMOTE_TRACE_SOURCE = "okahu"
REMOTE_SESSION_FACT = "session"
# Workflow the deployed agent exports under, when not passed to the constructor.
AGENTCORE_TRACE_WORKFLOW_ENV = "AGENTCORE_TRACE_WORKFLOW"

# AWS constrains the InvokeAgentRuntime runtimeSessionId header to 33-256 chars.
MIN_RUNTIME_SESSION_ID_LENGTH = 33
MAX_RUNTIME_SESSION_ID_LENGTH = 256

# Separator in the endpoint-qualified ARN form issued by the AgentCore console/CLI.
_ENDPOINT_MARKER = "/runtime-endpoint/"

# Header the retrieval key travels in. AgentCore only forwards a caller's headers
# to the agent under its own custom prefix, so the name the other runners send is
# nested inside it and unwrapped by the agent's span handler.
_RETRIEVAL_KEY_HEADER = AGENTCORE_CUSTOM_HEADER_PREFIX + TRACE_RETURN_REQUEST_HEADER
# Signed with the request, so the key cannot be swapped in transit.
_SIGN_EVENT = "before-sign.bedrock-agentcore.InvokeAgentRuntime"

DEFAULT_CONTENT_TYPE = "application/json"
_EVENT_STREAM_CONTENT_TYPE = "text/event-stream"
_SSE_DATA_PREFIX = "data: "


class AgentCoreRunner(AgentRunner):
    """Runner that invokes an agent already deployed to AWS Bedrock AgentCore Runtime.

    This is a remote runner: ``root_agent`` is the deployed agent's AgentCore
    Runtime ARN rather than an agent object, and the agent itself runs inside
    AWS. Only boto3 is needed locally, and it is imported lazily so the runner
    package stays importable without AWS dependencies installed.

    The request payload is ``{"prompt": <message>}``, matching what a
    ``BedrockAgentCoreApp`` entrypoint reads via ``payload.get("prompt")``.
    Pass a dict as the message to send a different shape verbatim.

    Spans are produced inside the deployed agent and exported by its own Monocle
    instrumentation rather than by this runner. They are retrieved from the
    trace backend by the session the agent was invoked with, so assertions see
    them alongside any local spans. That needs the workflow name the deployed
    agent reports under, via ``trace_workflow_name`` or the
    ``AGENTCORE_TRACE_WORKFLOW`` environment variable; without it retrieval is
    skipped and only the response is available to assert on.
    """

    def __init__(self, client: Any = None, region_name: Optional[str] = None,
                 trace_workflow_name: Optional[str] = None):
        """
        Args:
            client: Optional pre-built boto3 ``bedrock-agentcore`` client. When
                omitted, one is created lazily on first use.
            region_name: Optional AWS region for the lazily created client.
                When omitted, the region is taken from the runtime ARN.
            trace_workflow_name: Workflow name the deployed agent exports its
                spans under. Needed to retrieve them, since the deployed agent
                reports under its own workflow rather than the test's. Falls
                back to the ``AGENTCORE_TRACE_WORKFLOW`` environment variable;
                when neither is set the runner reports no remote trace source
                and retrieval is skipped.
        """
        self._client = client
        self._region_name = region_name
        self._trace_workflow_name = trace_workflow_name or os.environ.get(
            AGENTCORE_TRACE_WORKFLOW_ENV)
        self._last_session_id: Optional[str] = None
        self._remote_spans: list = []

    def _get_client(self, region_hint: Optional[str] = None) -> Any:
        """Return the boto3 client, creating it on first use.

        boto3 is imported here rather than at module scope to keep AgentCore an
        optional integration, as the other runners do with their frameworks.

        Region precedence: explicit constructor argument, then the region parsed
        from the runtime ARN, then boto3's ambient default. The ARN is preferred
        over the ambient default because an agent only exists in the region its
        ARN names, and reaching another region's endpoint fails as
        ResourceNotFoundException.
        """
        if self._client is None:
            try:
                import boto3
            except ImportError as e:
                raise ImportError(
                    "boto3 is required to use the AgentCore runner. "
                    "Install it with `pip install boto3`."
                ) from e
            region = self._region_name or region_hint
            kwargs = {"region_name": region} if region else {}
            self._client = boto3.client("bedrock-agentcore", **kwargs)
        return self._client

    @staticmethod
    def _region_from_arn(runtime_arn: str) -> Optional[str]:
        """Extract the region from an AgentCore Runtime ARN.

        ``arn:aws:bedrock-agentcore:<region>:<account>:runtime/<id>`` -> ``<region>``.
        Returns None for anything that isn't shaped like an ARN.
        """
        parts = runtime_arn.split(":")
        if len(parts) >= 4 and parts[0] == "arn" and parts[3]:
            return parts[3]
        return None

    @staticmethod
    def _parse_runtime_arn(root_agent: str) -> tuple[str, Optional[str]]:
        """Split an endpoint-qualified ARN into (runtime ARN, qualifier).

        The console and CLI issue the endpoint form
        ``arn:...:runtime/<id>/runtime-endpoint/DEFAULT``, while
        ``invoke_agent_runtime`` expects the runtime ARN with the endpoint name
        passed separately as ``qualifier``. Plain runtime ARNs are returned
        unchanged with no qualifier, so AWS applies the default endpoint.
        """
        if _ENDPOINT_MARKER in root_agent:
            runtime_arn, _, qualifier = root_agent.partition(_ENDPOINT_MARKER)
            return runtime_arn, (qualifier or None)
        return root_agent, None

    @staticmethod
    def _validate_session_id(session_id: str) -> None:
        """Reject a session id AgentCore cannot accept.

        The id is never padded or rewritten: the same value identifies the
        session in AWS and in the deployed agent's own traces, so substituting a
        different one would break that correspondence.
        """
        if len(session_id) < MIN_RUNTIME_SESSION_ID_LENGTH:
            raise ValueError(
                f"session_id {session_id!r} is {len(session_id)} characters, but AWS Bedrock "
                f"AgentCore requires runtimeSessionId to be at least "
                f"{MIN_RUNTIME_SESSION_ID_LENGTH} characters. Use a longer session id (for "
                f"example Monocle's auto-generated 'monocle_test_session_<uuid4 hex>'), or omit "
                f"session_id to let AgentCore generate one."
            )
        if len(session_id) > MAX_RUNTIME_SESSION_ID_LENGTH:
            raise ValueError(
                f"session_id is {len(session_id)} characters, but AWS Bedrock AgentCore allows "
                f"runtimeSessionId of at most {MAX_RUNTIME_SESSION_ID_LENGTH} characters."
            )

    @staticmethod
    def _build_payload(test_message: Union[str, dict, Any]) -> bytes:
        """Encode the test message into the deployed agent's request body."""
        if isinstance(test_message, dict):
            body = test_message
        else:
            body = {"prompt": test_message}
        return json.dumps(body).encode("utf-8")

    def _split_trailer(self, text: str) -> str:
        """Take any trace-return trailer off the response and keep its spans.

        A deployed agent running with trace-return enabled appends its spans to
        the value it returns, after a delimiter. Nothing announces that
        delimiter — boto3 surfaces no response header — so it is located by its
        fixed prefix. Returns the text with the trailer removed, so callers only
        ever see the agent's own answer.
        """
        index = text.find(_DELIMITER_PREFIX)
        if index == -1:
            return text
        end = text.find(_DELIMITER_SUFFIX, index + len(_DELIMITER_PREFIX))
        if end == -1:
            return text
        payload = text[end + len(_DELIMITER_SUFFIX):]
        try:
            from monocle_apptrace.instrumentation.common.trace_return import decode_payload
            self._remote_spans = JSONSpanLoader.from_json_str(decode_payload(payload))
        except Exception as e:
            logger.warning(f"Failed to deserialize spans returned by the agent: {e}")
            self._remote_spans = []
        return text[:index]

    def get_remote_spans(self) -> list:
        """Spans the deployed agent returned alongside its response, if any."""
        return self._remote_spans

    @staticmethod
    def _retrieval_key_handler():
        """A botocore hook that presents the trace-retrieval key on the request.

        The deployed agent returns its spans only to a caller that presents the
        key it was configured with. Returns None when
        ``MONOCLE_TRACE_RETRIEVAL_KEY`` is unset, so an unconfigured run sends
        nothing extra.
        """
        key = os.environ.get(MONOCLE_TRACE_RETRIEVAL_KEY_ENV)
        if not key:
            return None

        def add_retrieval_key(request, **_):
            already_set = {str(name).lower() for name in request.headers.keys()}
            if _RETRIEVAL_KEY_HEADER.lower() not in already_set:
                request.headers.add_header(_RETRIEVAL_KEY_HEADER, key)

        return add_retrieval_key

    def _invoke(self, client: Any, request: dict) -> dict:
        """Invoke the runtime, presenting the retrieval key when one is set.

        The hook is registered around this one call rather than on the client:
        the client may have been supplied by the caller and outlive the runner,
        and a key captured once at construction would go stale if the
        environment changed between tests.

        A client that takes no botocore hooks — a test double standing in for
        one — is still invoked, just without the key, since that is what the
        runner did before the key existed.
        """
        handler = self._retrieval_key_handler()
        events = getattr(getattr(client, "meta", None), "events", None)
        if handler is None or events is None:
            if handler is not None:
                logger.debug("client accepts no botocore hooks; retrieval key not sent")
            return client.invoke_agent_runtime(**request)
        events.register_first(_SIGN_EVENT, handler)
        try:
            return client.invoke_agent_runtime(**request)
        finally:
            events.unregister(_SIGN_EVENT, handler)

    @staticmethod
    def _decode_response(response: dict) -> Any:
        """Read and decode the AgentCore response stream.

        ``response["response"]`` is a botocore StreamingBody and has to be
        consumed. Server-sent-event responses are reassembled from their
        ``data:`` lines; everything else is read whole. The result is JSON
        decoded when possible, so an agent returning text yields a plain str and
        one returning an object yields a dict, and is returned as raw text
        otherwise.
        """
        stream = response.get("response")
        if stream is None:
            return None

        content_type = response.get("contentType") or ""
        if _EVENT_STREAM_CONTENT_TYPE in content_type:
            chunks = []
            for line in stream.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8")
                if line.startswith(_SSE_DATA_PREFIX):
                    line = line[len(_SSE_DATA_PREFIX):]
                chunks.append(line)
            text = "\n".join(chunks)
        else:
            raw = stream.read()
            text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)

        try:
            return json.loads(text)
        except (ValueError, TypeError):
            return text

    async def run_agent_async(self, root_agent: str, *args, session_id: str = None,
                              qualifier: str = None, **kwargs) -> Any:
        """Invoke a deployed AgentCore agent.

        Args:
            root_agent: The AgentCore Runtime ARN of the deployed agent. The
                endpoint-qualified form is accepted and split automatically.
            *args: The test message; the first positional argument is used.
            session_id: Monocle session id, sent as ``runtimeSessionId`` so the
                deployed agent keeps conversation context across turns. Omitted
                from the request when None, in which case AgentCore generates a
                session id and returns it.
            qualifier: Optional AgentCore endpoint name. Takes precedence over a
                qualifier embedded in ``root_agent``.
            **kwargs: Extra parameters passed through to ``invoke_agent_runtime``.

        Returns:
            The decoded agent response — a str for the common case of an agent
            returning text.
        """
        if root_agent is None or not isinstance(root_agent, str):
            raise ValueError(
                "For AgentCoreRunner, root_agent must be the AgentCore Runtime ARN string."
            )

        test_message = args[0] if args else None
        runtime_arn, arn_qualifier = self._parse_runtime_arn(root_agent)

        request = {
            "agentRuntimeArn": runtime_arn,
            "payload": self._build_payload(test_message),
            "contentType": DEFAULT_CONTENT_TYPE,
            "accept": DEFAULT_CONTENT_TYPE,
        }

        effective_qualifier = qualifier or arn_qualifier
        if effective_qualifier:
            request["qualifier"] = effective_qualifier

        if session_id is not None:
            self._validate_session_id(session_id)
            request["runtimeSessionId"] = session_id

        request.update(kwargs)

        # Remembered as sent, so remote spans are looked up by exactly the id
        # AgentCore stamped on them rather than a separately derived one.
        self._last_session_id = request.get("runtimeSessionId")
        # Spans belong to one invocation; a reused runner must not report the
        # previous call's.
        self._remote_spans = []

        # AWS errors are left to propagate so the framework's expect_errors
        # handling sees a real failure rather than an error string as a response.
        client = self._get_client(region_hint=self._region_from_arn(runtime_arn))
        response = self._invoke(client, request)
        logger.debug(f"AgentCore response statusCode={response.get('statusCode')}")
        result = self._decode_response(response)
        if isinstance(result, str):
            result = self._split_trailer(result)
        return result

    def get_remote_traces_source(self) -> Optional[str]:
        """Remote spans live in the trace backend the deployed agent exports to.

        Reports a source only once retrieval can actually succeed — a workflow
        name is configured and a session id was sent — so an unconfigured runner
        skips retrieval instead of quietly importing nothing.

        Reports nothing when the agent already returned its spans in the
        response: those are the same spans, so fetching them again would add
        every one of them twice.
        """
        if self._remote_spans:
            return None
        if self._trace_workflow_name and self._last_session_id:
            return REMOTE_TRACE_SOURCE
        return None

    def get_remote_trace_query(self) -> dict:
        """Identify the deployed agent's spans by the session it was invoked with.

        The agent records the ``runtimeSessionId`` it received as the
        ``agentic.session`` scope, which the trace backend indexes as the agent
        session fact, so the id sent on the call is enough to find the spans it
        produced.
        """
        if self._remote_spans or not (self._trace_workflow_name and self._last_session_id):
            return {}
        return {
            "id": self._last_session_id,
            "fact_name": REMOTE_SESSION_FACT,
            "workflow_name": self._trace_workflow_name,
        }

    def run_agent(self, root_agent: str, *args, session_id: str = None,
                  qualifier: str = None, **kwargs) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    self.run_agent_async(root_agent, *args, session_id=session_id,
                                         qualifier=qualifier, **kwargs),
                )
                return future.result()
        return asyncio.run(
            self.run_agent_async(root_agent, *args, session_id=session_id,
                                 qualifier=qualifier, **kwargs)
        )
