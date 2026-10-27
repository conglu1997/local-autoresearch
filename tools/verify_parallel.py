"""Replay a team run's admissions, prompts and decisions without executing candidates."""

import argparse
import json
import math
from pathlib import Path

from runner.core import candidate_source, feedback
from runner.evaluate import CPU_SECONDS, ROOT, WALL_SECONDS
from runner.parallel import prompt_for
from runner.teams import Experiment, Team, configuration
from tasks import TASKS
from tools.artifact_integrity import ArtifactIntegrity
from tools.final_protocol import verify_final_artifacts
from tools.verify_run import verify_stream


def verify_evaluation(result, task, seeds):
    if not result["valid"]:
        assert result["mean_score"] is None
        return
    assert [t["seed"] for t in result["trials"]] == seeds
    scores = []
    for trial in result["trials"]:
        assert trial["valid"] is True
        assert trial["exit_code"] == 0 and trial["timeout"] is False
        assert 0 <= trial["cpu_seconds"] <= CPU_SECONDS
        assert 0 <= trial["wall_seconds"] <= WALL_SECONDS
        solution = trial["circles"] if task == "circle" else trial["solution"]
        checked = TASKS[task].check(solution)
        assert math.isclose(checked["score"], trial["score"], rel_tol=0, abs_tol=1e-12)
        scores.append(checked["score"])
    assert math.isclose(
        math.fsum(scores) / len(scores), result["mean_score"], rel_tol=0, abs_tol=1e-12
    )


def verify(root):
    def read(path):
        return json.loads((root / path).read_text())

    manifest = read("manifest.json")
    config = manifest["parallel"]
    assert config == configuration(config["agents"], config["replicate"])
    integrity = ArtifactIntegrity(root)
    integrity.verify_files()
    task = manifest["task"]
    baseline = read("000/evaluation.json")
    verify_evaluation(baseline, task, manifest["seeds"])
    assert baseline["valid"]
    assert integrity.source_hash("000/solver.py") == baseline["source_sha256"]
    team = Team(
        Experiment("000", -1, baseline["mean_score"], True), config["agents"], config["replicate"]
    )
    ledger = [
        [{"attempt": "000", "decision": "baseline", **feedback(baseline)}]
        for _ in range(config["agents"])
    ]
    brief = (ROOT / f"tasks/{task}/program.md").read_text()
    active, admitted, finished = {}, set(), set()
    seen = {integrity.fingerprint("000/solver.py"): "000"}
    for line in (root / "journal.jsonl").read_text().splitlines():
        event = json.loads(line)
        name, agent = event["attempt"], event["agent"]
        directory = root / name
        if event["type"] == "admit":
            assert agent not in active and name not in admitted
            selection = read(f"{name}/admission.json")
            expected = team.select(agent)
            assert all(selection[k] == v for k, v in expected.items())
            assert selection["elapsed_seconds"] < manifest["budget_seconds"]
            request = read(f"{name}/request.json")
            assert request["model"] == manifest["model"]
            assert all(request[k] == v for k, v in manifest["sampling"].items())
            prompt = prompt_for(root, brief, selection, ledger[agent])
            if integrity.exported:
                assert len(request["messages"]) == 1 and request["messages"][0]["role"] == "user"
                assert (root / selection["parent"] / "solver.py").read_text() in request[
                    "messages"
                ][0]["content"]
            else:
                assert request["messages"] == [{"role": "user", "content": prompt}]
            active[agent] = name
            admitted.add(name)
        else:
            assert event["type"] == "complete" and active.pop(agent) == name
            result = read(f"{name}/evaluation.json")
            verify_evaluation(result, task, manifest["seeds"])
            source_path = directory / "solver.py"
            has_source = source_path.exists()
            if result["valid"]:
                assert has_source, f"{name}: valid evaluation has no solver"
            if has_source:
                assert integrity.source_hash(f"{name}/solver.py") == result["source_sha256"]
                response = read(f"{name}/response.json")
                assert source_path.read_text() == candidate_source(response)
            if (directory / "response.json").exists():
                verify_stream(directory)
            expected = team.observe(Experiment(name, agent, result["mean_score"], result["valid"]))
            decision = read(f"{name}/decision.json")
            assert all(decision[k] == v for k, v in expected.items())
            assert decision["parent"] == read(f"{name}/admission.json")["parent"]
            assert 0 <= decision["elapsed_seconds"] <= manifest["budget_seconds"]
            if has_source:
                identity = integrity.fingerprint(f"{name}/solver.py")
                assert decision["structural_sha256"] == identity
                if identity in seen:
                    assert decision["duplicate_of"] == seen[identity]
                else:
                    assert "duplicate_of" not in decision
                seen.setdefault(identity, name)
            ledger[agent].append(
                {"attempt": name, "decision": decision["decision"], **feedback(result)}
            )
            finished.add(name)
    summary = read("summary.json")
    assert summary["best_attempt"] == team.best.attempt
    assert summary["best_score"] == team.best.score
    assert summary["completed_experiments"] == len(finished)
    assert summary["incumbents"] == [e.attempt for e in team.incumbents]
    assert (root / "best.py").read_text() == (root / team.best.attempt / "solver.py").read_text()
    end = read("finished.json")
    assert end["completed_experiments"] == len(finished)
    assert end["admitted_experiments"] == len(admitted)
    final_state = verify_final_artifacts(
        root, manifest, team.best.attempt, len(finished), integrity
    )
    return {
        "completed": len(finished),
        "admitted": len(admitted),
        "unfinished": sorted(admitted - finished),
        "best": team.best.attempt,
        "score": team.best.score,
        "final_test": final_state,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(json.dumps(verify(parser.parse_args().run), indent=2))
