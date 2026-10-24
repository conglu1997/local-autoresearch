import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import research
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

    monkeypatch.setattr(research, "urlopen", lambda *args, **kwargs: Response())
    body = {"model": "flint", "messages": [{"role": "user", "content": "Try another idea."}]}
    token = "test-token-must-not-be-recorded"
    response = research.complete(body, "http://127.0.0.1:11500/v1", token, tmp_path)
    assert response["reasoning"] == "Consider a different layout."
    assert response["content"] == "Here is the candidate."
    assert response["usage"]["completion_tokens"] == 20
    assert response["finish_reason"] == "stop"
    assert response["first_token_seconds"] is not None
    assert json.loads((tmp_path / "request.json").read_text()) == body
    assert all(token not in path.read_text() for path in tmp_path.iterdir())


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.example.com/v1",
        "http://127.0.0.1.example.com/v1",
        "http://[2001:db8::1]/v1",
        "http://user:secret@localhost:8080/v1",
        "http://localhost:8080/v1?api_key=secret",
        "http://localhost:8080/v1#fragment",
        "https://user:secret@api.example.com/v1",
        "https://api.example.com/v1?api_key=secret",
        "https://api.example.com/v1#fragment",
        "https:///v1",
        "https://",
        "ftp://api.example.com/v1",
        "//api.example.com/v1",
    ],
)
def test_connection_rejects_insecure_or_incomplete_urls_and_embedded_credentials(base_url):
    with pytest.raises(ValueError, match="HTTPS base URL"):
        research.model_server(base_url, "test-model", None)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "[::1]"])
@pytest.mark.parametrize("scheme", ["http", "https"])
def test_connection_accepts_loopback_without_rec(host, scheme):
    url = f"{scheme}://{host}:8080/v1"
    config = research.model_server(url + "/", "test-model", None)
    assert config["endpoint"] == url
    assert research.api_key(config) is None


def test_connection_flags_must_be_paired():
    with pytest.raises(ValueError, match="requires --model"):
        research.model_server("http://localhost:8080/v1", None, None)
    with pytest.raises(ValueError, match="require --base-url"):
        research.model_server(None, "test-model", None)


@pytest.mark.parametrize(
    "base_url", ["https://api.example.com/v1", "https://models.example.net:8443/api/v1"]
)
@pytest.mark.parametrize("key_env", [None, "CUSTOM_MODEL_TEST_KEY"])
def test_https_connection_uses_only_the_explicit_credential_variable(
    base_url, key_env, monkeypatch
):
    monkeypatch.setenv("CUSTOM_MODEL_TEST_KEY", "test-credential")
    config = research.model_server(base_url + "/", "test-model", key_env)
    assert config == {
        "endpoint": base_url,
        "model": "test-model",
        "api_key_env": key_env,
    }
    assert research.api_key(config) == ("test-credential" if key_env else None)


@pytest.mark.parametrize("endpoint", ["http://127.0.0.1:8080/v1", "https://api.example.com/v1"])
@pytest.mark.parametrize("key_env", [None, "LOCAL_MODEL_TEST_KEY"])
def test_custom_model_run_resume_export_and_report(tmp_path, monkeypatch, key_env, endpoint):
    token = "private-test-credential"
    if key_env:
        monkeypatch.setenv(key_env, token)
    baseline = Path("tasks/route/baseline.py").read_text()
    chunk = {
        "choices": [{"delta": {"content": f"```python\n{baseline}```"}, "finish_reason": "stop"}]
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

    monkeypatch.setattr(research, "urlopen", respond)
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
    args.resume, args.experiments = True, 2
    research.run(args)
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
    config = {"api_key_env": "LOCAL_MODEL_TEST_KEY"}
    monkeypatch.delenv("LOCAL_MODEL_TEST_KEY", raising=False)
    with pytest.raises(KeyError):
        research.api_key(config)
    monkeypatch.setenv("LOCAL_MODEL_TEST_KEY", "")
    with pytest.raises(ValueError, match="must contain an API key"):
        research.api_key(config)
