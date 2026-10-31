import gzip
import io
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import HTTPHandler, HTTPSHandler, build_opener
from urllib.response import addinfourl

import pytest

import research
from runner import inference
from tools.export_run import export
from tools.verify_run import verify, verify_stream


@pytest.mark.parametrize("failure", ["http", "json"])
def test_transport_failures_keep_evidence_and_name_its_location(tmp_path, monkeypatch, failure):
    calls = []
    body = b'{"error":{"code":400,"message":"Inference request failed"}}'
    partial = b'data: {"choices":'

    def respond(request, **kwargs):
        calls.append(request)
        if failure == "http":
            raise HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))
        return io.BytesIO(partial)

    monkeypatch.setattr(inference.OPENER, "open", respond)
    with pytest.raises(HTTPError if failure == "http" else json.JSONDecodeError) as caught:
        inference.complete(
            {}, "https://api.example.com/v1", "test-credential", tmp_path, max_retries=0
        )
    assert len(calls) == 1
    assert str(tmp_path) in caught.value.__notes__[0]
    assert (tmp_path / "request.json").is_file()
    if failure == "http":
        assert (tmp_path / "http-error.txt").read_bytes() == body
    else:
        event = json.loads((tmp_path / "stream.jsonl").read_text())
        assert event["line"] == partial.decode()
    assert not (tmp_path / "response.json").exists()


@pytest.mark.parametrize("reasoning_field", ["reasoning", "reasoning_content"])
def test_local_stream_records_reasoning_content_and_usage_without_credentials(
    tmp_path, monkeypatch, reasoning_field
):
    chunks = [
        {
            "choices": [
                {"delta": {reasoning_field: "Consider a different layout."}, "finish_reason": None}
            ]
        },
        {"choices": [{"delta": {"content": "Here is the candidate."}, "finish_reason": None}]},
        {
            "choices": [{"delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    ]
    lines = [f"data: {json.dumps(chunk)}\n".encode() for chunk in chunks] + [b"data: [DONE]\n"]

    class Response:
        def __enter__(self):
            return iter(lines)

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(inference.OPENER, "open", lambda *args, **kwargs: Response())
    body = {"model": "flint", "messages": [{"role": "user", "content": "Try another idea."}]}
    token = "test-token-must-not-be-recorded"
    response = inference.complete(body, "http://127.0.0.1:11500/v1", token, tmp_path)
    assert response["reasoning"] == "Consider a different layout."
    assert response["content"] == "Here is the candidate."
    assert response["usage"]["completion_tokens"] == 20
    assert response["finish_reason"] == "stop"
    assert response["first_token_seconds"] is not None
    assert json.loads((tmp_path / "request.json").read_text()) == body
    assert all(token not in path.read_text() for path in tmp_path.iterdir())


def test_model_request_rejects_redirects(tmp_path, monkeypatch):
    def redirect(self, request):
        response = addinfourl(
            BytesIO(b"Moved"), {"location": "http://other.example/v1"}, request.full_url, 302
        )
        response.msg = "Found"
        return response

    monkeypatch.setattr(HTTPSHandler, "https_open", redirect)
    monkeypatch.setattr(
        HTTPHandler, "http_open", lambda *args: pytest.fail("API key forwarded to redirect target")
    )
    monkeypatch.setattr(inference, "OPENER", build_opener(inference.NoRedirects))
    with pytest.raises(HTTPError, match="302"):
        inference.complete({}, "https://api.example.com/v1", "private-test-credential", tmp_path)
    assert (tmp_path / "http-error.txt").read_text() == "Moved"


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.com/v1",
        "https://user:secret@api.example.com/v1",
        "https://api.example.com/v1?api_key=secret",
    ],
)
def test_connection_rejects_insecure_or_incomplete_urls_and_embedded_credentials(base_url):
    with pytest.raises(ValueError, match="HTTPS base URL"):
        inference.model_server(base_url, "test-model", None)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/v1",
        "http://localhost:8080/v1",
        "http://[::1]:8080/v1",
        "https://api.example.com/v1",
    ],
)
def test_connection_accepts_loopback_and_https_without_rec(url):
    config = inference.model_server(url + "/", "test-model", None)
    assert config["endpoint"] == url
    assert inference.api_key(config) is None


def test_connection_flags_must_be_paired():
    with pytest.raises(ValueError, match="requires --model"):
        inference.model_server("http://localhost:8080/v1", None, None)
    with pytest.raises(ValueError, match="require --base-url"):
        inference.model_server(None, "test-model", None)


@pytest.mark.parametrize("stream_error", [False, True])
def test_stream_preserves_usage_provenance_and_errors(tmp_path, monkeypatch, stream_error):
    content = "```python\ndef solve(problem, seed, budget):\n    return []\n```"
    chunks = [
        {
            "id": "gen-test",
            "model": "test-model",
            "provider": "Test Provider",
            "choices": [{"delta": {"content": content}, "finish_reason": "stop"}],
        },
    ]
    if stream_error:
        chunks.append({"error": {"code": 502, "message": "Provider disconnected"}})
    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0.001}
    chunks.append({"choices": [], "usage": usage})
    stream = ": PROCESSING\n\n"
    stream += "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks)
    stream += "data: [DONE]\n\n"
    monkeypatch.setattr(inference.OPENER, "open", lambda *a, **kw: io.BytesIO(stream.encode()))
    if stream_error:
        with pytest.raises(RuntimeError, match="Provider disconnected"):
            inference.complete(
                {}, "https://api.example.com/v1", "private-token", tmp_path, max_retries=0
            )
        result = json.loads((tmp_path / "response.json").read_text())
    else:
        result = inference.complete({}, "https://api.example.com/v1", "private-token", tmp_path)
    assert result["id"] == "gen-test"
    assert result["model"] == "test-model"
    assert result["provider"] == "Test Provider"
    assert result["usage"] == usage
    if stream_error:
        assert result["error"]["code"] == 502
        with pytest.raises(ValueError, match="did not finish: error"):
            research.candidate_source(result)
    else:
        assert research.candidate_source(result).startswith("def solve(")
    assert all("private-token" not in path.read_text() for path in tmp_path.iterdir())
    (tmp_path / "stream.jsonl.gz").write_bytes(
        gzip.compress((tmp_path / "stream.jsonl").read_bytes())
    )
    verify_stream(tmp_path)
    result["provider"] = "Changed Provider"
    (tmp_path / "response.json").write_text(json.dumps(result))
    with pytest.raises(AssertionError, match="provider"):
        verify_stream(tmp_path)


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:8080/v1", "https://api.example.com/v1"])
@pytest.mark.parametrize("key_env", [None, "MODEL_TEST_KEY"])
def test_custom_model_run_resume_export_and_report(tmp_path, monkeypatch, key_env, endpoint):
    token = "private-test-credential"
    if key_env:
        monkeypatch.setenv(key_env, token)
    baseline = Path("tasks/route/baseline.py").read_text()
    chunk = {
        "choices": [
            {"index": 0, "delta": {"content": f"```python\n{baseline}```"}, "finish_reason": "stop"}
        ]
    }
    requested = []

    class Response:
        def __enter__(self):
            return iter([f"data: {json.dumps(chunk)}\n".encode(), b"data: [DONE]\n"])

        def __exit__(self, *args):
            return False

    def respond(request, **kwargs):
        requested.append(json.loads(request.data))
        assert request.full_url == endpoint + "/chat/completions"
        assert request.get_header("Authorization") == (f"Bearer {token}" if key_env else None)
        return Response()

    monkeypatch.setattr(inference.OPENER, "open", respond)
    args = SimpleNamespace(
        run=tmp_path / "run",
        task="route",
        resume=False,
        experiments=1,
        max_tokens=1024,
        base_url=endpoint,
        model="test-model",
        api_key_env=key_env,
    )
    research.run(args)
    args.resume = True
    args.experiments = 2

    def interrupt(request, **kwargs):
        raise KeyboardInterrupt

    for _ in range(2):
        with monkeypatch.context() as patch:
            patch.setattr(inference.OPENER, "open", interrupt)
            with pytest.raises(KeyboardInterrupt):
                research.run(args)
        assert not (args.run / "002/decision.json").exists()
    research.run(args)
    assert not list(args.run.glob("002-interrupted-*"))
    retries = list((args.run / "002/retries").glob("*/request.json"))
    assert len(retries) == 2
    assert all(p.read_bytes() == (args.run / "002/request.json").read_bytes() for p in retries)
    assert [request["model"] for request in requested] == ["test-model", "test-model"]
    assert all("top_k" not in request for request in requested)
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["model"] == "test-model"
    assert manifest["inference"]["api_key_env"] == key_env
    assert manifest["sampling"]["max_tokens"] == 1024
    for name in ("001", "002"):
        decision = json.loads((args.run / name / "decision.json").read_text())
        assert decision == {"decision": "discard", "parent": "000"}
    args.model = "changed-model"
    with pytest.raises(RuntimeError, match="Model configuration changed"):
        research.run(args)
    target = tmp_path / "export"
    export(args.run, target)
    verify(target)
    document = (target / "report.html").read_text()
    assert "test-model" in document
    assert "Flint" not in document
    assert all(token not in path.read_text() for path in args.run.rglob("*") if path.is_file())


def test_missing_or_empty_custom_credential_fails(monkeypatch):
    config = {"api_key_env": "MODEL_TEST_KEY"}
    monkeypatch.delenv("MODEL_TEST_KEY", raising=False)
    with pytest.raises(KeyError):
        inference.api_key(config)
    monkeypatch.setenv("MODEL_TEST_KEY", "")
    with pytest.raises(ValueError, match="must contain an API key"):
        inference.api_key(config)
