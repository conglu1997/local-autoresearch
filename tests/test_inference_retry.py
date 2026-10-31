import io
import json
import ssl
from http.client import IncompleteRead
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

import research
from runner import inference
import research_agent as agent
from tools.export_run import export
from tools.verify_run import verify


def stream(*chunks):
    return io.BytesIO("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks).encode())


def terminal(finish="stop"):
    return {"choices": [{"delta": {"content": "complete"}, "finish_reason": finish}]}


@pytest.mark.parametrize(
    "failure",
    ["eof", "json", "connection", "timeout", "midstream", "tls", "tls_wrapped", 429, 502, 504],
)
def test_transient_failure_retries_identical_request_and_preserves_evidence(
    tmp_path, monkeypatch, failure
):
    requests, delays = [], []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        if len(requests) > 1:
            return stream(terminal())
        if failure == "eof":
            return stream({"choices": [{"delta": {"content": "partial"}}]})
        if failure == "json":
            return io.BytesIO(b'data: {"choices":')
        if failure == "connection":
            raise URLError("connection reset")
        if failure == "timeout":
            raise TimeoutError("read timed out")
        if failure in ("tls", "tls_wrapped"):
            error = ssl.SSLError(
                ssl.SSL_ERROR_SSL, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac"
            )
            raise URLError(error) if failure == "tls_wrapped" else error
        if failure == "midstream":

            class BrokenStream(io.BytesIO):
                def __iter__(self):
                    yield b'data: {"choices": [{"delta": {"content": "partial"}}]}\n'
                    raise IncompleteRead(b"truncated")

            return BrokenStream()
        raise HTTPError(request.full_url, failure, "temporary failure", {}, io.BytesIO(b"busy"))

    monkeypatch.setattr(inference.OPENER, "open", respond)
    monkeypatch.setattr(inference.time, "sleep", delays.append)
    result = inference.complete({"messages": []}, "https://example.com/v1", "private-key", tmp_path)
    assert result["content"] == "complete"
    assert requests == [{"messages": []}] * 2
    assert delays == [30]
    archive = tmp_path / "retries/001"
    assert json.loads((archive / "request.json").read_text()) == requests[0]
    assert json.loads((archive / "retry.json").read_text())["reason"]
    assert all("private-key" not in p.read_text() for p in tmp_path.rglob("*") if p.is_file())


@pytest.mark.parametrize("wrapped", [False, True])
def test_certificate_verification_failure_is_recorded_without_retry(tmp_path, monkeypatch, wrapped):
    requests = []
    cause = ssl.SSLCertVerificationError(
        1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
    )
    error = URLError(cause) if wrapped else cause

    def respond(request, **kwargs):
        requests.append(request)
        raise error

    monkeypatch.setattr(inference.OPENER, "open", respond)
    monkeypatch.setattr(inference.time, "sleep", lambda seconds: pytest.fail("Must not retry"))
    with pytest.raises(type(error), match="CERTIFICATE_VERIFY_FAILED"):
        inference.complete({}, "https://example.com/v1", None, tmp_path)
    assert len(requests) == 1
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert not failure["retryable"] and failure["attempts_this_call"] == 1
    assert "CERTIFICATE_VERIFY_FAILED" in failure["error_summary"]
    assert (tmp_path / "request.json").exists()
    assert not (tmp_path / "retries").exists()


def test_tls_retry_budget_preserves_last_request_and_diagnostics(tmp_path, monkeypatch):
    requests, delays = [], []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        raise ssl.SSLError(ssl.SSL_ERROR_SSL, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac")

    monkeypatch.setattr(inference.OPENER, "open", respond)
    monkeypatch.setattr(inference.time, "sleep", delays.append)
    with pytest.raises(ssl.SSLError, match="SSLV3_ALERT_BAD_RECORD_MAC") as caught:
        inference.complete(
            {"messages": []}, "https://example.com/v1", None, tmp_path, max_retries=2
        )
    assert requests == [{"messages": []}] * 3 and delays == [30, 60]
    failure = json.loads((tmp_path / "failure.json").read_text())
    assert failure["retryable"] and failure["attempts_this_call"] == 3
    assert "SSLError" in failure["error_summary"]
    assert sorted(p.name for p in (tmp_path / "retries").iterdir()) == ["001", "002"]
    assert (tmp_path / "request.json").exists()
    assert any("request and any partial stream saved" in note for note in caught.value.__notes__)
    assert any("Failed after 3 attempts" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("code", [408, 429, 500, 502, 503, 504, "504"])
def test_streamed_provider_error_is_retried(tmp_path, monkeypatch, code):
    responses = iter(
        [stream({"error": {"code": code, "message": "temporary"}}), stream(terminal())]
    )
    monkeypatch.setattr(inference.OPENER, "open", lambda *a, **kw: next(responses))
    monkeypatch.setattr(inference.time, "sleep", lambda seconds: None)
    assert (
        inference.complete({}, "https://example.com/v1", None, tmp_path)["finish_reason"] == "stop"
    )
    saved = json.loads((tmp_path / "retries/001/response.json").read_text())
    assert saved["error"]["code"] == code


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404])
def test_permanent_http_error_is_not_retried(tmp_path, monkeypatch, code):
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        raise HTTPError(request.full_url, code, "permanent", {}, io.BytesIO(b"invalid"))

    monkeypatch.setattr(inference.OPENER, "open", respond)
    with pytest.raises(HTTPError):
        inference.complete({}, "https://example.com/v1", None, tmp_path)
    assert len(requests) == 1
    assert not (tmp_path / "retries").exists()


@pytest.mark.parametrize("finish", ["stop", "tool_calls", "length", "content_filter"])
def test_terminal_response_needs_no_done_marker_or_transport_retry(tmp_path, monkeypatch, finish):
    requests = []

    def respond(request, **kwargs):
        requests.append(request)
        return stream(terminal(finish))

    monkeypatch.setattr(inference.OPENER, "open", respond)
    assert (
        inference.complete({}, "https://example.com/v1", None, tmp_path)["finish_reason"] == finish
    )
    assert len(requests) == 1


def test_permanent_streamed_error_is_not_retried(tmp_path, monkeypatch):
    responses = iter([stream({"error": {"code": 401, "message": "invalid key"}})])
    monkeypatch.setattr(inference.OPENER, "open", lambda *a, **kw: next(responses))
    with pytest.raises(RuntimeError, match="invalid key"):
        inference.complete({}, "https://example.com/v1", None, tmp_path)
    assert json.loads((tmp_path / "response.json").read_text())["error"]["code"] == 401
    assert not json.loads((tmp_path / "failure.json").read_text())["retryable"]
    assert not (tmp_path / "retries").exists()


def test_retry_budget_is_bounded_and_final_partial_response_is_preserved(tmp_path, monkeypatch):
    requests, delays = [], []

    def respond(request, **kwargs):
        requests.append(request)
        return stream({"choices": [{"delta": {"content": "partial"}}]})

    monkeypatch.setattr(inference.OPENER, "open", respond)
    monkeypatch.setattr(inference.time, "sleep", delays.append)
    with pytest.raises(RuntimeError, match="checkpoint saved"):
        inference.complete({}, "https://example.com/v1", None, tmp_path, max_retries=2)
    result = json.loads((tmp_path / "response.json").read_text())
    assert result["finish_reason"] is None
    assert len(requests) == 3 and delays == [30, 60]
    assert sorted(p.name for p in (tmp_path / "retries").iterdir()) == ["001", "002"]
    assert json.loads((tmp_path / "response.json").read_text()) == result


@pytest.mark.parametrize("failure", ["eof", "tls", "tls_eof"])
def test_interrupted_submit_is_not_executed_and_retry_exports_and_replays(
    tmp_path, monkeypatch, failure
):
    requests, evaluations = [], []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        first = len(requests) == 1
        call = {
            "index": 0,
            "id": "submit",
            "type": "function",
            "function": {
                "name": "submit",
                "arguments": json.dumps(
                    {
                        "summary": "must not execute" if first else "submitted after retry",
                    }
                ),
            },
        }
        response = stream(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [call]},
                        "finish_reason": None if first else "tool_calls",
                    }
                ]
            }
        )
        if first and failure in ("tls", "tls_eof"):

            class BrokenTLSStream(io.BytesIO):
                def __iter__(self):
                    yield from self.getvalue().splitlines(keepends=True)
                    if failure == "tls_eof":
                        raise ssl.SSLEOFError(
                            ssl.SSL_ERROR_EOF, "EOF occurred in violation of protocol"
                        )
                    raise ssl.SSLError(
                        ssl.SSL_ERROR_SSL, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac"
                    )

            return BrokenTLSStream(response.getvalue())
        return response

    evaluate = research.evaluate

    def counted_evaluate(*args):
        evaluations.append(args[0])
        return evaluate(*args)

    monkeypatch.setattr(inference.OPENER, "open", respond)
    monkeypatch.setattr(inference.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(research, "evaluate", counted_evaluate)
    monkeypatch.setattr(agent, "evaluate", counted_evaluate)
    args = SimpleNamespace(
        run=tmp_path / "run",
        task="circle",
        experiments=1,
        resume=False,
        max_tokens=1024,
        max_steps=1,
        base_url="https://example.com/v1",
        model="test-tool-model",
        api_key_env=None,
    )
    research.run(args, agent=agent.propose)
    assert len(requests) == 2 and requests[0] == requests[1]
    assert len(evaluations) == 2  # Baseline, then the accepted submission exactly once.
    submission = json.loads((args.run / "001/submission.json").read_text())
    assert submission == {"summary": "submitted after retry"}
    assert [p.name for p in (args.run / "001/calls").iterdir()] == ["001"]
    target = tmp_path / "export"
    export(args.run, target)
    verify(target)
    assert (target / "001/calls/001/retries/001/stream.jsonl.gz").is_file()

    def interrupted_backoff(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(inference.time, "sleep", interrupted_backoff)
    monkeypatch.setattr(
        inference.OPENER,
        "open",
        lambda *a, **kw: stream({"choices": [{"delta": {"content": "unfinished"}}]}),
    )
    args.resume, args.experiments = True, 2
    with pytest.raises(KeyboardInterrupt):
        research.run(args, agent=agent.propose)
    assert len(evaluations) == 2
    assert not (args.run / "002/calls/001/request.json").exists()
    interrupted = tmp_path / "interrupted-export"
    export(args.run, interrupted)
    verify(interrupted)
    assert (interrupted / "002/calls/001/retries/001/request.json").is_file()
    monkeypatch.setattr(inference.OPENER, "open", respond)
    monkeypatch.setattr(inference.time, "sleep", lambda seconds: None)
    research.run(args, agent=agent.propose)
    assert len(requests) == 3 and len(evaluations) == 3
    assert json.loads((args.run / "summary.json").read_text())["completed_experiments"] == 2
    resumed = tmp_path / "resumed-export"
    export(args.run, resumed)
    verify(resumed)


def tool_response(*actions):
    calls = [
        {
            "index": i,
            "id": f"call-{i}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }
        for i, (name, args) in enumerate(actions)
    ]
    return stream({"choices": [{"delta": {"tool_calls": calls}, "finish_reason": "tool_calls"}]})


@pytest.fixture
def recovery_run(tmp_path, monkeypatch):
    from pathlib import Path
    from runner.evaluate import digest
    from tasks.circle import check
    from tasks.circle.baseline import solve

    evaluations = []
    baseline = Path("tasks/circle/baseline.py").read_text()

    def evaluate(source, seeds, task):
        assert source in (baseline, baseline + "\n# changed\n")
        evaluations.append(source)
        return {
            "valid": True,
            "mean_score": check(solve(26, 0, 15))["score"],
            "source_sha256": digest(source),
            "trials": [{"valid": True, "seed": 0, **check(solve(26, 0, 15))}],
        }

    monkeypatch.setattr(research, "evaluate", evaluate)
    monkeypatch.setattr(agent, "evaluate", evaluate)
    monkeypatch.setattr(inference.time, "sleep", lambda seconds: None)
    args = SimpleNamespace(
        run=tmp_path / "run",
        task="circle",
        experiments=1,
        resume=False,
        max_tokens=1024,
        max_steps=4,
        base_url="https://example.com/v1",
        model="test-tool-model",
        api_key_env=None,
    )
    return args, evaluations, baseline


@pytest.mark.parametrize("failure", ["http", "stream", "eof", "tls"])
def test_inference_failures_resume_same_round_without_repeating_completed_calls(
    tmp_path, monkeypatch, recovery_run, failure
):
    args, evaluations, baseline = recovery_run
    requests = []
    accepting = False

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        if len(requests) == 1:
            return tool_response(("edit_solver", {"source": baseline + "\n# changed\n"}))
        if accepting:
            return tool_response(("submit", {"summary": "Resume the saved draft."}))
        if failure == "http":
            raise HTTPError(
                request.full_url, 429, "quota exhausted", {}, io.BytesIO(b"quota exhausted")
            )
        if failure == "stream":
            return stream({"error": {"code": 504, "message": "temporary outage"}})
        if failure == "tls":
            raise ssl.SSLError(
                ssl.SSL_ERROR_SSL, "[SSL: SSLV3_ALERT_BAD_RECORD_MAC] bad record mac"
            )
        return stream({"choices": [{"delta": {"content": "incomplete"}}]})

    monkeypatch.setattr(inference.OPENER, "open", respond)
    with pytest.raises((HTTPError, RuntimeError, ssl.SSLError)):
        research.run(args, agent=agent.propose)
    assert len(requests) == 7 and len(evaluations) == 1
    assert not (args.run / "001/decision.json").exists()
    assert not (args.run / "001/evaluation.json").exists()
    assert (args.run / "001/draft.py").read_text() == baseline + "\n# changed\n"
    assert json.loads((args.run / "summary.json").read_text())["completed_experiments"] == 0
    prefix = {p: p.read_bytes() for p in (args.run / "001/calls/001").iterdir()}
    retry_evidence = {
        p: p.read_bytes() for p in (args.run / "001/calls/002/retries").rglob("*") if p.is_file()
    }
    assert (
        json.loads((args.run / "001/calls/002/failure.json").read_text())["failure_kind"]
        == "inference_error"
    )
    paused_export = tmp_path / "paused-export"
    export(args.run, paused_export)
    verify(paused_export)

    args.resume = True
    with pytest.raises((HTTPError, RuntimeError, ssl.SSLError)):
        research.run(args, agent=agent.propose)
    assert len(requests) == 13 and len(evaluations) == 1
    assert all(p.read_bytes() == data for p, data in retry_evidence.items())
    accepting = True
    research.run(args, agent=agent.propose)
    assert len(requests) == 14 and len(evaluations) == 2
    assert all(request == requests[1] for request in requests[2:])
    assert all(p.read_bytes() == data for p, data in prefix.items())
    assert not (args.run / "001/calls/002/failure.json").exists()
    assert len(list((args.run / "001/calls/002/retries").iterdir())) == 12
    assert json.loads((args.run / "summary.json").read_text())["completed_experiments"] == 1
    final_export = tmp_path / "resumed-export"
    export(args.run, final_export)
    verify(final_export)


@pytest.mark.parametrize("interruption", ["after_response", "after_evaluation"])
@pytest.mark.parametrize("mode", ["agent", "direct"])
def test_resume_reuses_saved_response_and_evaluation(
    tmp_path, monkeypatch, recovery_run, interruption, mode
):
    args, evaluations, baseline = recovery_run
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        if mode == "direct":
            return stream(
                {
                    "choices": [
                        {
                            "delta": {"content": f"```python\n{baseline}\n# changed\n```"},
                            "finish_reason": "stop",
                        }
                    ]
                }
            )
        return tool_response(
            ("edit_solver", {"source": baseline + "\n# changed\n"}),
            ("submit", {"summary": "Record the changed draft."}),
        )

    monkeypatch.setattr(inference.OPENER, "open", respond)
    processor, field = (agent, "advance") if mode == "agent" else (research, "candidate_source")
    process, save = getattr(processor, field), research.save
    if interruption == "after_response":

        def interrupt(*args):
            raise KeyboardInterrupt

        monkeypatch.setattr(processor, field, interrupt)
    else:

        def interrupt(path, value):
            if path.name == "decision.json" and path.parent.name == "001":
                raise KeyboardInterrupt
            return save(path, value)

        monkeypatch.setattr(research, "save", interrupt)
    with pytest.raises(KeyboardInterrupt):
        research.run(args, agent=agent.propose if mode == "agent" else None)
    assert len(requests) == 1
    assert len(evaluations) == (1 if interruption == "after_response" else 2)
    monkeypatch.setattr(processor, field, process)
    monkeypatch.setattr(research, "save", save)
    args.resume = True
    research.run(args, agent=agent.propose if mode == "agent" else None)
    assert len(requests) == 1 and len(evaluations) == 2
    target = tmp_path / "export"
    export(args.run, target)
    verify(target)


def test_resumed_call_rejects_changed_messages_before_requesting(tmp_path, monkeypatch):
    research.save(tmp_path / "request.json", {"messages": []})
    monkeypatch.setattr(inference.OPENER, "open", lambda *a, **kw: pytest.fail("Network called"))
    with pytest.raises(RuntimeError, match="Saved request differs"):
        inference.complete(
            {"messages": [{"role": "user", "content": "changed"}]},
            "https://example.com/v1",
            None,
            tmp_path,
        )


def test_resume_request_mismatch_keeps_saved_draft_and_conversation(monkeypatch, recovery_run):
    args, evaluations, baseline = recovery_run
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        if len(requests) == 1:
            return tool_response(("edit_solver", {"source": baseline + "\n# changed\n"}))
        raise KeyboardInterrupt

    monkeypatch.setattr(inference.OPENER, "open", respond)
    with pytest.raises(KeyboardInterrupt):
        research.run(args, agent=agent.propose)
    checkpoint = {p: p.read_bytes() for p in (args.run / "001").iterdir() if p.is_file()}
    first_request = args.run / "001/calls/001/request.json"
    changed = json.loads(first_request.read_text())
    changed["messages"][1]["content"] += " altered"
    research.save(first_request, changed)
    args.resume = True
    with pytest.raises(RuntimeError, match="Saved request differs"):
        research.run(args, agent=agent.propose)
    assert len(requests) == 2 and len(evaluations) == 1
    assert all(p.read_bytes() == value for p, value in checkpoint.items())
