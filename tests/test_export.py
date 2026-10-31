import gzip
import hashlib
import json
from pathlib import Path

import pytest

from tools import export_run

MANIFEST = json.dumps({"inference": {"api_key_env": "TEST_KEY"}})


@pytest.mark.parametrize("sse", [False, True])
def test_export_redacts_paths_and_preserves_stream(tmp_path, monkeypatch, sse):
    monkeypatch.setenv("TEST_KEY", "k" * 32)
    monkeypatch.setattr(export_run, "report", lambda root: (root / "report.html").write_text(""))
    source, target = tmp_path / "source", tmp_path / "target"
    attempt = source / "001"
    attempt.mkdir(parents=True)
    paired_baseline = '{"valid": false, "trials": []}\n'
    (source / "baseline-final-test.json").write_text(paired_baseline)
    message = f"File {Path.home()}/code/example.py failed"
    if sse:
        message = "data: " + json.dumps({"choices": [{"index": 0, "delta": {"content": message}}]})
    raw = json.dumps({"seconds": 0, "line": message}) + "\n"
    (source / "manifest.json").write_text(MANIFEST)
    (attempt / "stream.jsonl").write_text(raw)
    (attempt / "diagnostic.json").write_text(raw)
    (attempt / "http-error.txt").write_text(raw)
    # A configurable agent budget can exceed three digits.
    for number in ("001", "1000"):
        step = attempt / "calls" / number
        step.mkdir(parents=True)
        (step / "request.json").write_text(raw)
        (step / "stream.jsonl").write_text(raw)
    export_run.export(source, target)
    exported = gzip.decompress((target / "001/stream.jsonl.gz").read_bytes()).decode()
    assert exported == raw.replace(str(Path.home()), "<HOME>")
    assert (target / "001/diagnostic.json").read_text() == exported
    assert (target / "001/http-error.txt").read_text() == exported
    for number in ("001", "1000"):
        step = target / "001/calls" / number
        assert (step / "request.json").read_text() == exported
        assert gzip.decompress((step / "stream.jsonl.gz").read_bytes()).decode() == exported
    assert (target / "baseline-final-test.json").read_text() == paired_baseline
    record = next(
        entry
        for entry in json.loads((target / "export.json").read_text())["files"]
        if entry["file"] == "001/stream.jsonl.gz"
    )
    assert record["original_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert record["export_sha256"] == hashlib.sha256(exported.encode()).hexdigest()


def test_export_refuses_local_token(tmp_path, monkeypatch):
    token = "secret-for-test-1234567890"
    monkeypatch.setenv("TEST_KEY", token)
    source = tmp_path / "source"
    source.mkdir()
    (source / "manifest.json").write_text(MANIFEST)
    (source / "summary.json").write_text(json.dumps({"leaked": token}))
    with pytest.raises(RuntimeError, match="Credential detected"):
        export_run.export(source, tmp_path / "target")


@pytest.mark.parametrize("field", ["content", "reasoning", "tool_calls"])
def test_export_rejects_credential_split_across_unfinished_stream(tmp_path, monkeypatch, field):
    key = "synthetic-local-credential-12345"
    monkeypatch.setenv("TEST_KEY", key)
    monkeypatch.setattr(export_run, "report", lambda root: pytest.fail("Export must stop"))
    source, target = tmp_path / "source", tmp_path / "target"
    attempt = source / "001"
    attempt.mkdir(parents=True)
    (source / "manifest.json").write_text(MANIFEST)
    events = []
    for index, fragment in enumerate(key):
        delta = (
            {field: fragment}
            if field != "tool_calls"
            else {"tool_calls": [{"index": 0, "function": {"arguments": fragment}}]}
        )
        chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
        events.append(json.dumps({"seconds": index, "line": "data: " + json.dumps(chunk)}))
    (attempt / "stream.jsonl").write_text("\n".join(events) + "\n")
    with pytest.raises(RuntimeError, match="Credential detected in stream.jsonl"):
        export_run.export(source, target)
    assert not (target / "export.json").exists()


def test_redacts_mounted_volume_paths_but_keeps_filenames():
    text = json.dumps(
        {
            "absolute": "/Volumes/scratch/user/tmp/solver.py",
            "relative": "Volumes/scratch/user/tmp/solver.py",
            "traceback": "Warning\n/Volumes/scratch/user/scipy/model.py:12\n",
            "local": "./solver.py",
        }
    )
    assert json.loads(export_run.redact(text)) == {
        "absolute": "<ABSOLUTE_PATH>/solver.py",
        "relative": "<RELATIVE_PATH>/solver.py",
        "traceback": "Warning\n<ABSOLUTE_PATH>/model.py:12\n",
        "local": "./solver.py",
    }


def test_stream_redaction_handles_paths_split_across_chunks():
    original = 'open("/Volumes/scratch/user/solver.py")'
    lines = []
    for index, char in enumerate(original):
        chunk = {"choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}]}
        lines.append(json.dumps({"seconds": index, "line": "data: " + json.dumps(chunk)}) + "\n")
    redacted = [
        json.loads(json.loads(line)["line"][6:]) for line in export_run.redacted_stream(lines)
    ]
    assert len(redacted) == len(lines)
    assembled = "".join(chunk["choices"][0]["delta"]["content"] for chunk in redacted)
    assert assembled == 'open("<ABSOLUTE_PATH>/solver.py")'


def test_stream_keeps_server_errors():
    error = {"error": {"code": 500, "message": "Compute error.", "type": "server_error"}}
    lines = [
        json.dumps({"seconds": 0.2, "line": "data: " + json.dumps(error)}) + "\n",
        json.dumps({"seconds": 0.3, "line": "data: [DONE]"}) + "\n",
    ]
    assert list(export_run.redacted_stream(lines)) == lines


def test_tool_argument_redaction_handles_paths_split_across_chunks():
    original = json.dumps({"source": 'open("/Volumes/scratch/user/solver.py")'})
    lines = []
    for index, char in enumerate(original):
        delta = {"tool_calls": [{"index": 0, "function": {"arguments": char}}]}
        chunk = {"choices": [{"index": 0, "delta": delta}]}
        lines.append(json.dumps({"seconds": index, "line": "data: " + json.dumps(chunk)}) + "\n")
    redacted = [
        json.loads(json.loads(line)["line"][6:]) for line in export_run.redacted_stream(lines)
    ]
    assembled = "".join(
        chunk["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"] for chunk in redacted
    )
    assert json.loads(assembled) == {"source": 'open("<ABSOLUTE_PATH>/solver.py")'}
