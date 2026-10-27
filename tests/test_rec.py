"""A serial search through rec records, resumes and keeps the token out of the run."""

import json
from types import SimpleNamespace

import research
from runner import inference


def test_rec_run_records_and_resumes(tmp_path, monkeypatch):
    endpoint = "http://localhost:11501/v1"
    token = "test-credential-never-recorded"
    status = {
        "ready": True,
        "model": "flint",
        "endpoint": endpoint,
        "engine": "test-rec-engine",
        "version": "test-rec-version",
    }

    def rec_command(argv, **kwargs):
        assert argv in (["rec", "status"], ["rec", "auth", "token"])
        return json.dumps(status) if argv == ["rec", "status"] else token

    monkeypatch.setattr(inference.subprocess, "check_output", rec_command)
    source = "def solve(seed):\n    return 'synthetic candidate'\n"
    chunk = {
        "choices": [{"delta": {"content": f"```python\n{source}```"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }
    requests = []

    class Response:
        def __enter__(self):
            return iter([f"data: {json.dumps(chunk)}\n".encode(), b"data: [DONE]\n"])

        def __exit__(self, *args):
            return False

    def respond(request, **kwargs):
        assert request.full_url == endpoint + "/chat/completions"
        assert request.get_header("Authorization") == "Bearer " + token
        requests.append(json.loads(request.data))
        return Response()

    monkeypatch.setattr(inference.OPENER, "open", respond)
    evaluated = []

    def evaluate(candidate, seeds, task):
        assert seeds == [0] and task == "circle"
        evaluated.append(candidate)
        return {"valid": True, "mean_score": len(evaluated)}

    monkeypatch.setattr(research, "evaluate", evaluate)
    args = SimpleNamespace(
        run=tmp_path / "run",
        task="circle",
        resume=False,
        experiments=1,
        max_tokens=1024,
        base_url=None,
        model=None,
        api_key_env=None,
    )
    research.run(args)
    first = {p.name: p.read_bytes() for p in (args.run / "001").iterdir()}
    args.resume, args.experiments = True, 2
    research.run(args)
    assert first == {p.name: p.read_bytes() for p in (args.run / "001").iterdir()}
    assert evaluated[1:] == [source, source]
    assert len(requests) == 2
    assert all(r["temperature"] == 1 and r["top_p"] == 0.95 for r in requests)
    assert all(r["top_k"] == 20 for r in requests)
    assert json.loads((args.run / "summary.json").read_text())["completed_experiments"] == 2
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert {"runner/core.py", "runner/inference.py"} <= manifest["fixed_files"].keys()
    for number in ["001", "002"]:
        record = args.run / number
        assert (record / "solver.py").read_text() == source
        assert {
            "request.json",
            "response.json",
            "response.txt",
            "stream.jsonl",
            "solver.py",
            "evaluation.json",
            "decision.json",
        } == {p.name for p in record.iterdir()}
    assert all(token not in p.read_text() for p in args.run.rglob("*") if p.is_file())
