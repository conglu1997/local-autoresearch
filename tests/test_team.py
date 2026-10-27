"""Team selection and prompts, and real workers against a fake streaming model server."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import count
import json
import threading

import pytest

import research
from report import report
from runner import parallel
from runner.core import save
from runner.parallel import prompt_for
from runner.teams import Experiment, Team
from runner.evaluate import ROOT
from tools import export_run, verify_parallel

BASELINE = (ROOT / "tasks/circle/baseline.py").read_text()
BETTER = """def solve(n, seed, budget):
    circles = [[(c + 0.5) / 5, (r + 0.5) / 5, 0.1 - 1e-12] for r in range(5) for c in range(5)]
    return circles + [[0.2, 0.2, 0.04]]
"""


def stream(*chunks):
    return b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)


def answer(content):
    return stream(
        {
            "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
        }
    )


REPLIES = [
    answer(f"```python\n{BETTER}```"),
    answer(f"```python\n# Same grid.\n{BASELINE}```"),
    answer("```python\ndef solve(:\n```"),
    stream({"error": {"code": 500, "message": "Compute error.", "type": "server_error"}}),
]


def serve(replies):
    class Model(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(replies[next(calls) % len(replies)] + b"data: [DONE]\n\n")

        def log_message(self, *args):
            pass

    calls = count()
    server = ThreadingHTTPServer(("127.0.0.1", 0), Model)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def run_team(root, replies, hours, workers=8):
    server = serve(replies)
    url = f"http://127.0.0.1:{server.server_port}/v1"
    argv = ["--task", "circle", "--run", str(root), "--hours", str(hours)]
    argv += ["--workers", str(workers)]
    research.main([*argv, "--base-url", url, "--model", "flint"])
    server.shutdown()
    return json.loads((root / "finished.json").read_text())


def test_team_run_verifies_reports_and_exports(tmp_path):
    root = tmp_path / "run"
    assert run_team(root, REPLIES, 8 / 3600)["reason"] == "time budget"
    sampling = json.loads((root / "manifest.json").read_text())["sampling"]
    assert sampling == {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}
    result = verify_parallel.verify(root)
    assert result["completed"] >= 4
    assert result["score"] == pytest.approx(2.54, abs=1e-9)
    assert (root / "best.py").read_text() == BETTER
    decisions = [json.loads(p.read_text()) for p in root.glob("[0-9]*/decision.json")]
    errors = [json.loads(p.read_text()).get("error") for p in root.glob("[0-9]*/evaluation.json")]
    assert any(d.get("duplicate_of") == "000" for d in decisions)
    assert "Inference failed: Compute error." in errors

    report(root)
    assert "circle · 8 workers" in (root / "report.html").read_text()
    export_run.export(root, tmp_path / "export")
    assert verify_parallel.verify(tmp_path / "export")["score"] == result["score"]


def test_crashing_workers_are_recorded_then_retired(tmp_path, monkeypatch):
    monkeypatch.setattr(parallel, "RETRY_SECONDS", 0)
    root = tmp_path / "run"
    finished = run_team(root, [b"data: not json\n\n"], 60 / 3600)
    assert finished["reason"] == "every worker failed three times in a row"
    assert finished["completed_experiments"] == 24
    assert verify_parallel.verify(root)["best"] == "000"
    evaluation = json.loads((root / "001/evaluation.json").read_text())
    assert evaluation["error"] == "Worker exited 1; see worker.log"
    assert "JSONDecodeError" in (root / "001/worker.log").read_text()


def test_one_worker_runs_the_same_loop(tmp_path):
    root = tmp_path / "run"
    run_team(root, [answer(f"```python\n{BETTER}```")], 6 / 3600, workers=1)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["parallel"]["agents"] == 1
    result = verify_parallel.verify(root)
    assert result["completed"] >= 2
    assert result["score"] == pytest.approx(2.54, abs=1e-9)


def baseline():
    return Experiment("000", -1, 1.0, True)


def test_every_request_starts_from_the_best_program():
    team = Team(baseline())
    for i in range(8):
        assert team.select(i)["parent"] == "000"
    team.observe(Experiment("003", 2, 3.0, True))
    assert all(team.select(i)["parent"] == "003" for i in range(8))
    before = list(team.incumbents)
    team.observe(Experiment("009", 7, None, False))
    assert team.incumbents == before and team.best.attempt == "003"
    assert team.select(7)["parent"] == "003"


def test_directions_rotate_per_worker():
    team = Team(baseline(), replicate=1)
    assert [team.select(0)["request_variant"] for _ in range(5)] == [1, 2, 3, 0, 1]


def test_ties_and_weaker_late_completions_do_not_replace_winner():
    team = Team(baseline())
    team.observe(Experiment("002", 1, 3.0, True))
    team.select(0)
    assert team.observe(Experiment("001", 0, 2.0, True))["decision"] == "discard"
    assert not team.observe(Experiment("003", 2, 3.0, True))["global_improvement"]
    assert team.best.attempt == "002"


def test_prompt_contains_winning_code_and_worker_feedback(tmp_path):
    for name, source in [("000", "def solve(): return 0\n"), ("003", "def solve(): return 3\n")]:
        (tmp_path / name).mkdir()
        (tmp_path / name / "solver.py").write_text(source)
        save(tmp_path / name / "evaluation.json", {"valid": True, "mean_score": 3})
    team = Team(baseline())
    team.observe(Experiment("003", 2, 3.0, True))
    prompt = prompt_for(
        tmp_path,
        "Task",
        team.select(0),
        [{"attempt": "000", "decision": "baseline", "error": "worker detail"}],
    )
    assert "def solve(): return 3" in prompt and "worker detail" in prompt
    assert "Refine the current method" in prompt


def test_failed_json_write_leaves_previous_record_readable(tmp_path):
    path = tmp_path / "summary.json"
    save(path, {"completed": 4})
    with pytest.raises(TypeError):
        save(path, {"unsupported": object()})
    assert json.loads(path.read_text()) == {"completed": 4}
