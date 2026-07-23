from monocle_apptrace.instrumentation.common import trace_return as tr


# ---- default callback ----

def test_default_callback_match(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.default_trace_retrieval_callback({"x-monocle-retrieve-traces": "s3cret"}) is True


def test_default_callback_mismatch(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.default_trace_retrieval_callback({"x-monocle-retrieve-traces": "wrong"}) is False


def test_default_callback_header_absent(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.default_trace_retrieval_callback({"other": "s3cret"}) is False


def test_default_callback_key_unset(monkeypatch):
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", raising=False)
    assert tr.default_trace_retrieval_callback({"x-monocle-retrieve-traces": "anything"}) is False


def test_default_callback_case_insensitive_header(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.default_trace_retrieval_callback({"X-Monocle-Retrieve-Traces": "s3cret"}) is True


# ---- dispatch ----

def test_authorized_uses_default_when_no_custom(monkeypatch):
    monkeypatch.delenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", raising=False)
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_DEFAULT_KEY", "s3cret")
    assert tr.is_trace_return_authorized({"x-monocle-retrieve-traces": "s3cret"}) is True
    assert tr.is_trace_return_authorized({"x-monocle-retrieve-traces": "no"}) is False


def test_authorized_custom_callback(tmp_path, monkeypatch):
    mod = tmp_path / "cb_ok.py"
    mod.write_text("def cb(headers):\n    return headers.get('x-ok') == 'yes'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "cb_ok:cb")
    assert tr.is_trace_return_authorized({"x-ok": "yes"}) is True
    assert tr.is_trace_return_authorized({"x-ok": "no"}) is False


def test_authorized_unresolvable_callback_denies(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "no_such_module_xyz:cb")
    assert tr.is_trace_return_authorized({"x-monocle-retrieve-traces": "x"}) is False


def test_authorized_not_callable_denies(tmp_path, monkeypatch):
    mod = tmp_path / "cb_notcallable.py"
    mod.write_text("NOT_A_FUNC = 42\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "cb_notcallable:NOT_A_FUNC")
    assert tr.is_trace_return_authorized({}) is False


def test_authorized_callback_raises_denies(tmp_path, monkeypatch):
    mod = tmp_path / "cb_boom.py"
    mod.write_text("def cb(headers):\n    raise RuntimeError('boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "cb_boom:cb")
    assert tr.is_trace_return_authorized({}) is False


def test_authorized_malformed_spec_denies(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "no_colon_here")
    assert tr.is_trace_return_authorized({}) is False


def test_authorized_empty_module_spec_denies(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", ":cb")
    assert tr.is_trace_return_authorized({}) is False


def test_authorized_relative_module_spec_denies(monkeypatch):
    monkeypatch.setenv("MONOCLE_TRACE_RETRIEVAL_CALLBACK", "..bad:cb")
    assert tr.is_trace_return_authorized({}) is False
