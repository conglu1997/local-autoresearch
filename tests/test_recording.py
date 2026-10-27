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
from tools.verify_run import verify


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
    for _ in range(2):
        (args.run / "002").mkdir()
        (args.run / "002/request.json").write_text("{}")
        research.run(args)
    args.experiments = 2
    research.run(args)
    interrupted = sorted(p.name for p in args.run.glob("002-*"))
    assert interrupted == ["002-interrupted-1", "002-interrupted-2"]
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
