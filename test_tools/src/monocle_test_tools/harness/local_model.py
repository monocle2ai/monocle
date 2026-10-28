"""``model="local"``: a small open model on this machine, so scenario tests need no API key.

Needs ``llama-cpp-python``; the error says how to install it. On first use the model is
downloaded into the Hugging Face cache and llama.cpp's OpenAI-compatible server starts as a
child process.
"""
import atexit
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.request
from typing import Any, Optional

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

QWEN = ("bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF", "Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
MODELS = {                                   # name -> (Hugging Face repo, 4-bit GGUF file)
    "local": QWEN,
    "local:qwen3-4b": QWEN,
    "local:phi-4-mini": ("bartowski/microsoft_Phi-4-mini-instruct-GGUF", "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"),
    "local:llama-3.2-3b": ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
}
WHEEL_INDEX = "https://abetlen.github.io/llama-cpp-python/whl/"
RUNTIME_VERSION = "0.3.19"                    # newest release with prebuilt wheels
CONTEXT_TOKENS = 8192
STARTUP_SECONDS = 180

_server: Optional[subprocess.Popen] = None
_port: Optional[int] = None
_model_path: Optional[str] = None


def is_local_spec(spec: Any) -> bool:
    return isinstance(spec, str) and (spec == "local" or spec.startswith("local:"))


def local_chat_model(spec: str = "local") -> Any:
    if spec not in MODELS:
        raise ValueError(f"unknown local model '{spec}'; known: {', '.join(MODELS)}")
    try:
        import llama_cpp.server  # noqa: F401
    except ImportError:
        raise ImportError("model='local' needs llama-cpp-python. Install it with "
                          f"`{install_command()}`") from None
    path = ensure_model(*MODELS[spec])
    port = ensure_server(path)
    return LocalChatOpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="local",
                           model=os.path.basename(path), temperature=0, timeout=600)


def install_command() -> str:
    """Pinned to the newest release with prebuilt wheels: later PyPI releases are source only and
    need a compiler. Metal wheels stop at Python 3.12; later Pythons get the CPU wheel."""
    metal = sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info < (3, 13)
    return (f'pip install "llama-cpp-python[server]=={RUNTIME_VERSION}" --only-binary llama-cpp-python '
            f"--extra-index-url {WHEEL_INDEX}{'metal' if metal else 'cpu'}")


def ensure_model(repo: str, filename: str) -> str:
    from huggingface_hub import hf_hub_download, try_to_load_from_cache

    cached = try_to_load_from_cache(repo, filename)
    if isinstance(cached, str):
        return cached
    logger.info("downloading %s/%s (one time, about 2.5 GB)", repo, filename)
    return hf_hub_download(repo, filename)


def ensure_server(model_path: str) -> int:
    """Start the server once per process and return its port. One server, one model."""
    global _server, _port, _model_path
    if _server is not None and _server.poll() is None:
        if model_path != _model_path:
            raise ValueError(f"a local model is already running ({os.path.basename(_model_path)}); "
                             "use the same local model for the test agent and the judge")
        return _port
    _model_path = model_path
    _port = _free_port()
    _server = subprocess.Popen(
        [sys.executable, "-m", "llama_cpp.server", "--model", model_path, "--n_gpu_layers", "-1",
         "--n_ctx", str(CONTEXT_TOKENS), "--host", "127.0.0.1", "--port", str(_port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    atexit.register(stop_server)
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        if _server.poll() is not None:
            raise RuntimeError(f"the local model server exited with code {_server.returncode}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{_port}/v1/models", timeout=2)
            return _port
        except Exception:  # not up yet
            time.sleep(1)
    stop_server()
    raise RuntimeError(f"the local model server did not come up within {STARTUP_SECONDS}s")


def stop_server() -> None:
    global _server
    if _server is not None and _server.poll() is None:
        _server.terminate()
        try:
            _server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _server.kill()
    _server = None


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


TOOL_TAGS = re.compile(r"</?tool_call>|<\|/?tool_call\|>|<\|python_tag\|>")


def lift_tool_calls(message: Any, tool_names: set) -> None:
    """The server returns a model's tool calls as text, and each model writes them its own way:
    Qwen as ``<tool_call>{...}</tool_call>``, Phi as a JSON list, Llama as a bare
    ``{"name": ..., "parameters": ...}``. Any JSON naming an offered tool becomes a tool call."""
    text = message.content if isinstance(message.content, str) else ""
    if not text or not tool_names or message.tool_calls:
        return
    found, spans = [], []
    for value, start, end in _json_values(text):
        calls = [c for c in _calls(value) if c[0] in tool_names]
        if calls:
            found.extend(calls)
            spans.append((start, end))
    if not found:
        return
    for index, (name, args) in enumerate(found):
        message.tool_calls.append({"name": name, "args": args, "id": f"call_{id(message)}_{index}",
                                   "type": "tool_call"})
    for start, end in reversed(spans):
        text = text[:start] + text[end:]
    message.content = TOOL_TAGS.sub("", text).strip()


def _json_values(text: str):
    decoder, index = json.JSONDecoder(), 0
    while True:
        starts = [i for i in (text.find("{", index), text.find("[", index)) if i >= 0]
        if not starts:
            return
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue
        yield value, start, end
        index = end


def _calls(value: Any):
    if isinstance(value, list):
        for item in value:
            yield from _calls(item)
        return
    if not isinstance(value, dict):
        return
    if isinstance(value.get("function"), dict):
        value = value["function"]
    args = value.get("arguments", value.get("parameters"))
    if not isinstance(value.get("name"), str) or args is None:
        return
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return
    if isinstance(args, dict):
        yield value["name"], args


def _tool_names(kwargs: dict) -> set:
    return {t.get("function", {}).get("name") for t in kwargs.get("tools") or () if isinstance(t, dict)}


class LocalChatOpenAI(ChatOpenAI):
    """Lifts tool calls out of the text; structured output uses a forced tool call, the only
    structured-output path the server supports."""

    def _generate(self, *args, **kwargs):
        result = super()._generate(*args, **kwargs)
        for generation in result.generations:
            lift_tool_calls(generation.message, _tool_names(kwargs))
        return result

    async def _agenerate(self, *args, **kwargs):
        result = await super()._agenerate(*args, **kwargs)
        for generation in result.generations:
            lift_tool_calls(generation.message, _tool_names(kwargs))
        return result

    def _get_request_payload(self, *args, **kwargs):
        # llama.cpp's server requires a string `content` even on an assistant message
        # that only carries tool calls; OpenAI clients send null there.
        payload = super()._get_request_payload(*args, **kwargs)
        for message in payload.get("messages", []):
            if message.get("content") is None:
                message["content"] = ""
        return payload

    def with_structured_output(self, schema, **kwargs):
        kwargs["method"] = "function_calling"
        return super().with_structured_output(schema, **kwargs)
