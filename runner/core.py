"""Candidate extraction, prompts and records shared by serial and team search."""

import ast
import json
import re


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def failure_info(result, default="evaluation_error"):
    if result.get("valid"):
        return {"failure_kind": None, "error_summary": None}
    failure = next((trial for trial in result.get("trials", []) if not trial["valid"]), result)
    return {
        "failure_kind": failure.get("failure_kind", result.get("failure_kind", default)),
        "error_summary": (failure.get("error") or result.get("error") or "Evaluation failed")[:500],
    }


def feedback(result):
    if "trials" not in result:
        return result
    return {
        "valid": result["valid"],
        "mean_score": result["mean_score"],
        **failure_info(result),
        "trials": [
            {
                key: value
                for key, value in trial.items()
                if key
                not in ("circles", "certificate", "solution", "correlations", "failed_output")
            }
            for trial in result["trials"]
        ],
    }


def candidate_source(response):
    if response["finish_reason"] != "stop":
        raise ValueError(f"Model response did not finish: {response['finish_reason']}")
    blocks = re.findall(r"```python\s*\n(.*?)```", response["content"], re.DOTALL)
    if len(blocks) != 1:
        raise ValueError("Expected exactly one complete Python code block")
    source = blocks[0].strip() + "\n"
    ast.parse(source)
    return source


def previous_attempt(best, ledger):
    previous = best.parent / ledger[-1]["attempt"]
    if previous == best or not (previous / "solver.py").exists():
        return ""
    return (
        f"\nPrevious attempt ({ledger[-1]['decision']}):\n```python\n"
        + (previous / "solver.py").read_text()
        + "```\n"
    )


def make_prompt(brief, best, ledger):
    return (
        brief
        + "\nCurrent best solver:\n```python\n"
        + (best / "solver.py").read_text()
        + "```\nRecent results:\n"
        + json.dumps(ledger[-4:], indent=2)
        + previous_attempt(best, ledger)
        + "\nWhat would you try next?"
    )


def make_team_prompt(brief, best, ledger):
    return (
        brief
        + "\nStarting solver:\n```python\n"
        + (best / "solver.py").read_text()
        + "```\nStarting solver result:\n"
        + json.dumps(feedback(json.loads((best / "evaluation.json").read_text())))
        + "\nRecent results:\n"
        + json.dumps(ledger[-4:], indent=2)
        + previous_attempt(best, ledger)
        + "\nWhat would you try next?"
    )
