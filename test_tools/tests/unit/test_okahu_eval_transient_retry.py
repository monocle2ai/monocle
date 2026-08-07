"""Transient transport failures must not cost a scenario -- but a retry must never
create a duplicate eval job.

Observed on a 3-run x 20-scenario replay matrix: 3 of 60 evals were lost to
`ConnectionResetError` and a fact-map read timeout. None was retried:

  * `get_fact_map()` had no retry at all.
  * the eval-submit POST retried only `requests.Timeout`, so a connection reset
    (a `RequestException`) fell straight through.

Retrying is only safe where the request is idempotent. Submitting an eval job is NOT:
each POST creates a new job id (and bills for it), so a retry after the request may
already have been delivered can double-charge. Hence two policies:

  * idempotent reads (fact map, template listing) -- retry any transport failure.
  * the submit POST -- retry ONLY `requests.ConnectTimeout`, where the connection was
    never established and nothing can have reached the service. This deliberately
    NARROWS the previous behaviour, which retried read timeouts too.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from monocle_test_tools.evals.okahu_eval import OkahuEval

MODULE = "monocle_test_tools.evals.okahu_eval"
TEMPLATE = {"name": "t", "eval_prompt": "x",
            "structure_output": {"label": {"enums": ["a"], "description": "x"},
                                 "explanation": {"description": "x"}}}
JUDGE = {"label": "a", "explanation": "why", "total_tokens": 7}


def _json_response(payload):
    m = MagicMock()
    m.headers = {"Content-Type": "application/json"}
    m.raise_for_status.return_value = None
    m.json.return_value = payload
    return m


def _eval(monkeypatch):
    monkeypatch.setenv("OKAHU_API_KEY", "k")
    return OkahuEval(eval_options={"trace_source": "okahu"})


def _span():
    span = MagicMock()
    span.attributes = {"workflow.name": "wf"}
    span.start_time = 1_000_000_000
    span.end_time = 2_000_000_000
    return span


def _submit(ev, post_side_effect):
    """Drive evaluate() far enough to hit the submit POST, isolating it from the
    fact-map and export paths."""
    with patch.object(OkahuEval, "export_trace", return_value="traceid"), \
         patch.object(OkahuEval, "enumerate_fact_ids", return_value=["traceid"]), \
         patch(f"{MODULE}.OkahuEvalResultExporter"), \
         patch(f"{MODULE}.time.sleep") as sleep, \
         patch(f"{MODULE}.requests.post", side_effect=post_side_effect) as post:
        try:
            result = ev.evaluate(filtered_spans=[_span()], template=TEMPLATE, fact_name="traces")
        except AssertionError as exc:
            return post, sleep, exc
        return post, sleep, result


# --- idempotent reads: retry freely -----------------------------------------------

def test_fact_map_retries_a_connection_reset_then_succeeds(monkeypatch):
    ev = _eval(monkeypatch)
    effects = [requests.ConnectionError("Connection aborted"),
               _json_response({"traces": "trace_id"})]

    with patch(f"{MODULE}.requests.get", side_effect=effects) as get, \
         patch(f"{MODULE}.time.sleep") as sleep:
        assert ev.get_fact_map() == {"traces": "trace_id"}

    assert get.call_count == 2, "a reset on an idempotent read must be retried"
    assert sleep.called, "retries must back off rather than hammer the service"


def test_fact_map_retries_a_read_timeout_then_succeeds(monkeypatch):
    # Safe here precisely because the request is a read: re-issuing it cannot create
    # state or a charge.
    ev = _eval(monkeypatch)
    effects = [requests.ReadTimeout("read timeout=60"), _json_response({"traces": "trace_id"})]

    with patch(f"{MODULE}.requests.get", side_effect=effects) as get, \
         patch(f"{MODULE}.time.sleep"):
        assert ev.get_fact_map() == {"traces": "trace_id"}

    assert get.call_count == 2


def test_fact_map_gives_up_after_the_attempt_budget(monkeypatch):
    ev = _eval(monkeypatch)

    with patch(f"{MODULE}.requests.get",
               side_effect=requests.ConnectionError("Connection aborted")) as get, \
         patch(f"{MODULE}.time.sleep"):
        with pytest.raises(AssertionError, match="Failed to reach fact map service"):
            ev.get_fact_map()

    assert get.call_count == 3, "should stop at the attempt budget, not loop forever"


def test_backoff_grows_between_attempts(monkeypatch):
    ev = _eval(monkeypatch)

    with patch(f"{MODULE}.requests.get",
               side_effect=requests.ConnectionError("boom")), \
         patch(f"{MODULE}.time.sleep") as sleep:
        with pytest.raises(AssertionError):
            ev.get_fact_map()

    delays = [c.args[0] for c in sleep.call_args_list]
    assert delays == sorted(delays) and len(set(delays)) > 1, (
        f"expected increasing backoff, got {delays}")


def test_successful_read_does_not_sleep(monkeypatch):
    ev = _eval(monkeypatch)

    with patch(f"{MODULE}.requests.get", return_value=_json_response({"traces": "trace_id"})), \
         patch(f"{MODULE}.time.sleep") as sleep:
        ev.get_fact_map()

    sleep.assert_not_called()


# --- the submit POST: retry only when nothing could have been delivered -----------

def test_submit_retries_a_connect_timeout_then_succeeds(monkeypatch):
    # ConnectTimeout means the connection was never established, so no job exists yet.
    ev = _eval(monkeypatch)
    effects = [requests.ConnectTimeout("connect timed out"),
               _json_response({"job_id": "j1", "result": [{"result": json.dumps(JUDGE)}]})]

    post, sleep, result = _submit(ev, effects)

    assert post.call_count == 2
    assert result == ("a", "why")


def test_submit_does_not_retry_a_connection_reset(monkeypatch):
    # The request may already have reached the service; a retry could create a second
    # billable eval job. Fail instead, and say why.
    ev = _eval(monkeypatch)

    post, sleep, exc = _submit(ev, requests.ConnectionError("Connection aborted"))

    assert post.call_count == 1, "must not resubmit a possibly-delivered eval job"
    assert isinstance(exc, AssertionError)
    assert "duplicate" in str(exc).lower(), (
        f"the error must explain why no retry happened, got: {exc}")


def test_submit_does_not_retry_a_read_timeout(monkeypatch):
    # Deliberate narrowing: the previous implementation retried read timeouts, but the
    # request WAS sent, so that risked duplicate jobs.
    ev = _eval(monkeypatch)

    post, sleep, exc = _submit(ev, requests.ReadTimeout("read timeout=120"))

    assert post.call_count == 1
    assert isinstance(exc, AssertionError)
    assert "duplicate" in str(exc).lower()


def test_submit_gives_up_after_the_attempt_budget_on_connect_timeout(monkeypatch):
    ev = _eval(monkeypatch)

    post, sleep, exc = _submit(ev, requests.ConnectTimeout("connect timed out"))

    assert post.call_count == 3
    assert isinstance(exc, AssertionError)
    assert "connect" in str(exc).lower()
