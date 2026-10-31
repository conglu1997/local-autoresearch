import gzip
import io
import json
import difflib
from pathlib import Path
from types import SimpleNamespace

import pytest

import research
from runner import inference
import research_agent as agent
from runner.evaluate import digest
from tasks import TASKS
from tools.export_run import export
from tools.verify_run import inventory, verify, verify_agent, verify_stream


def call(name, args=None, identifier="call-1"):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args or {})},
    }


def response(calls, finish="tool_calls"):
    return {
        "content": "",
        "reasoning": "Plan an edit.",
        "tool_calls": calls,
        "finish_reason": finish,
    }


def stream(calls, finish="tool_calls"):
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "reasoning": "Plan an edit.",
                        "reasoning_details": [{"type": "reasoning.text", "text": "Plan an edit."}],
                    }
                }
            ]
        }
    ]
    for index, item in enumerate(calls):
        arguments = item["function"]["arguments"]
        split = len(arguments) // 2
        chunks.append(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    **item,
                                    "index": index,
                                    "function": {
                                        **item["function"],
                                        "arguments": arguments[:split],
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        )
        chunks.append(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": index,
                                    "function": {"arguments": arguments[split:]},
                                }
                            ]
                        }
                    }
                ]
            }
        )
    chunks.append({"choices": [{"delta": {}, "finish_reason": finish}]})
    chunks.append(
        {
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30, "cost": 0},
        }
    )
    return io.BytesIO(
        ("".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n").encode()
    )


def test_stream_reassembles_tool_arguments_and_preserves_reasoning(tmp_path, monkeypatch):
    calls = [call("read_solver"), call("edit_solver", {"patch": "@@\n-a\n-b\n+c"}, "call-2")]
    monkeypatch.setattr(inference.OPENER, "open", lambda *a, **kw: stream(calls))
    result = inference.complete(
        {"tools": agent.TOOLS}, "https://api.example.com/v1", "secret", tmp_path
    )
    assert result["tool_calls"] == calls
    assert result["reasoning_details"][0]["text"] == "Plan an edit."
    (tmp_path / "stream.jsonl.gz").write_bytes(
        gzip.compress((tmp_path / "stream.jsonl").read_bytes())
    )
    verify_stream(tmp_path)
    result["tool_calls"][1]["function"]["arguments"] = "{}"
    research.save(tmp_path / "response.json", result)
    with pytest.raises(AssertionError):
        verify_stream(tmp_path)


def test_plain_reasoning_fragments_round_trip_as_one_string(tmp_path):
    recorded = response([call("read_solver")])
    recorded["reasoning_details"] = [
        {"type": "reasoning.text", "text": piece, "format": "unknown", "index": 0}
        for piece in ("Plan", " an", " edit.")
    ]
    workspace = agent.Workspace(tmp_path, "x = 1\n", [])
    messages = []
    assert agent.advance(recorded, workspace, messages) is None
    assert messages[0]["reasoning"] == "Plan an edit."
    assert "reasoning_details" not in messages[0]
    assert len(recorded["reasoning_details"]) == 3  # Original evidence stays untouched.


@pytest.mark.parametrize("kind", ["signed", "encrypted", "identified"])
def test_structured_reasoning_keeps_identity_and_signatures(tmp_path, kind):
    recorded = response([call("read_solver")])
    detail = {"type": "reasoning.text", "text": recorded["reasoning"], "index": 0}
    if kind == "signed":
        detail["signature"] = "preserve-this-signature"
    elif kind == "encrypted":
        detail = {"type": "reasoning.encrypted", "data": "opaque", "index": 0}
    else:
        detail["id"] = "reasoning-id"
    recorded["reasoning_details"] = [detail]
    messages = []
    assert agent.advance(recorded, agent.Workspace(tmp_path, "x = 1\n", []), messages) is None
    assert messages[0]["reasoning_details"] == [detail]
    assert "reasoning" not in messages[0]


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_attempt", {"attempt": "../../.env"}),
        ("read_attempt", {"attempt": "999"}),
        ("read_attempt", {"attempt": 0}),
        ("load_solver", {"attempt": "../../.env"}),
        ("load_solver", {"attempt": "999"}),
        ("load_solver", {"attempt": 3}),
        ("load_solver", {"attempt": "000", "path": "solver.py"}),
        ("__init__", {}),
        ("read_solver", {"path": "~/.env"}),
        ("list_attempts", {"limit": 100}),
        ("list_attempts", {"offset": -1}),
        ("list_attempts", {"limit": True}),
        ("list_attempts", {"order": "oldest"}),
        ("search_attempts", {"query": ""}),
        ("search_attempts", {"query": "x", "field": "files"}),
        ("search_attempts", {"query": "x", "limit": True}),
        ("edit_solver", {}),
        ("edit_solver", {"patch": "", "source": "x"}),
        ("edit_solver", {"patch": "@@\n-missing\n+x"}),
        ("edit_solver", {"source": None}),
        ("edit_solver", {"old": "aaa = 'a'\n", "new": "x"}),
    ],
)
def test_tools_constrain_history_and_edits(tmp_path, name, args):
    workspace = agent.Workspace(tmp_path, "aaa = 'a'\n", [])
    assert "error" in workspace.dispatch(call(name, args))
    assert workspace.source == "aaa = 'a'\n"
    assert not workspace.submitted
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "name,args,details",
    [
        (
            "submit",
            {"new": "Keep the solver."},
            [
                "missing required arguments: summary",
                "unexpected arguments: new",
                "allowed arguments: summary",
            ],
        ),
        (
            "submit",
            {"summary": 42, "description": "Keep the solver."},
            [
                "unexpected arguments: description",
                "allowed arguments: summary",
                "summary must be string, got int",
            ],
        ),
        ("submit", {"summary": None}, ["summary must be string, got NoneType"]),
        (
            "read_solver",
            {"path": "solver.py"},
            ["unexpected arguments: path", "allowed arguments: start_line, end_line"],
        ),
        ("list_attempts", {"limit": True}, ["limit must be integer, got bool"]),
        ("list_attempts", {"offset": -1}, ["offset must be >= 0", "received -1", "Use 0"]),
        ("list_attempts", {"limit": 0}, ["limit must be between 1 and 50", "received 0"]),
        ("list_attempts", {"limit": 51}, ["limit must be between 1 and 50", "received 51"]),
        (
            "list_attempts",
            {"order": "oldest"},
            ["order must be one of", "recent", "score", "received 'oldest'"],
        ),
        (
            "search_attempts",
            {"query": "x", "field": "files"},
            ["field must be one of", "summary", "code", "received 'files'"],
        ),
        (
            "search_attempts",
            {"query": "x", "limit": 100},
            ["limit must be between 1 and 50", "received 100"],
        ),
        (
            "search_attempts",
            {"query": " \t "},
            ["query must be nonblank", "whitespace-only", "Provide a keyword or phrase"],
        ),
        (
            "search_attempts",
            {"query": "x\ny"},
            ["query must be a single-line", "remove line breaks"],
        ),
        (
            "read_attempt",
            {"attempt": "999"},
            ["Unknown previous attempt '999'", "list_attempts", "including leading zeros"],
        ),
        (
            "load_solver",
            {"version": "003"},
            ["missing required arguments: attempt", "unexpected arguments: version"],
        ),
        (
            "edit_solver",
            {"patch": "@@\n-absent\n+x"},
            ["Patch hunk 1 (patch line 1)", "context not found", "read_solver"],
        ),
        (
            "edit_solver",
            {"patch": "@@\n+x", "source": "x"},
            ["exactly one of patch", "source", "received both fields"],
        ),
        ("edit_solver", {}, ["exactly one of patch", "source", "received neither field"]),
        (
            "submit",
            {"summary": "a" * 501},
            ["received 501 characters", "maximum 500", "1 over", "Shorten summary"],
        ),
        (
            "submit",
            {"summary": "改" * 520},
            ["received 520 characters", "maximum 500", "20 over", "Shorten summary"],
        ),
        (
            "submit",
            {"summary": " " + "改" * 500},
            ["received 501 characters", "maximum 500", "1 over", "count includes whitespace"],
        ),
        (
            "submit",
            {"summary": ""},
            ["summary must be nonblank", "received 0 characters", "maximum 500", "Write a brief"],
        ),
        (
            "submit",
            {"summary": " \n\t "},
            ["summary must be nonblank", "received 4 characters", "whitespace-only", "maximum 500"],
        ),
    ],
)
def test_tool_errors_explain_how_to_correct_arguments(tmp_path, name, args, details):
    workspace = agent.Workspace(tmp_path, "aaa = 'a'\n", [])
    result = workspace.dispatch(call(name, args))
    assert result["error"].startswith(name + ":")
    assert all(detail in result["error"] for detail in details)
    assert "This call did not change the draft or submit it" in result["error"]
    assert "the round is still open" in result["error"]
    parameters = result["expected_arguments"]
    assert parameters["additionalProperties"] is False
    if name == "submit":
        assert parameters["required"] == ["summary"]
        assert parameters["properties"]["summary"]["maxLength"] == 500
    assert workspace.source == "aaa = 'a'\n" and not workspace.submitted
    assert workspace.summary is None


def test_syntax_error_can_be_repaired_and_submit_freezes_code(tmp_path):
    workspace = agent.Workspace(tmp_path, "return (", [])
    result = workspace.dispatch(call("submit", {"summary": "Repair the solver."}))
    assert "SyntaxError" in result["error"]
    assert "solver.py line 1, column 8" in result["error"]
    assert "return (" in result["error"]
    assert "Fix the draft with edit_solver, then retry submit" in result["error"]
    assert "no code was executed" in result["error"]
    assert not workspace.submitted
    assert workspace.summary is None
    repaired = "def solve(n, seed, budget):\n    return []\n"
    messages = []
    assert (
        agent.advance(
            response(
                [
                    call("edit_solver", {"source": repaired}),
                    call("submit", {"summary": "Repair the solver."}, "call-2"),
                    call("edit_solver", {"source": "x = 3\n"}, "call-3"),
                ]
            ),
            workspace,
            messages,
        )
        is None
    )
    assert workspace.submitted and workspace.source == repaired
    assert workspace.summary == "Repair the solver."
    assert "already submitted" in messages[-1]["content"]


@pytest.mark.parametrize(
    "source,detail",
    [
        ("return []\n", "outside function"),
        ("break\n", "outside loop"),
        ("continue\n", "not properly in loop"),
        ("nonlocal missing\n", "module level"),
        ("x = 1\0\n", "null bytes"),
    ],
)
def test_compile_errors_leave_submission_open_for_repair(tmp_path, source, detail):
    workspace = agent.Workspace(tmp_path, source, [])
    result = workspace.dispatch(call("submit", {"summary": "Check this draft."}))
    assert "SyntaxError" in result["error"] and detail in result["error"]
    assert "solver.py" in result["error"] and "None" not in result["error"]
    if "\0" not in source:
        assert "solver.py line 1" in result["error"]
    assert not workspace.submitted and workspace.summary is None
    # Compiling a repaired draft must not execute its top-level code.
    repaired = "raise RuntimeError('must only run in the evaluator')\n"
    assert workspace.dispatch(call("edit_solver", {"source": repaired})) == {"edited": True}
    assert workspace.dispatch(call("submit", {"summary": "Check compilation only."}))["submitted"]


def history_workspace(root):
    records = [
        ("000", -5, "baseline", "x = 0\n", None),
        ("001", -1, "keep", "radius = 1\n", "Tune the Radius"),
        ("002", None, "invalid", "raise RuntimeError()\n", "Try an optimizer"),
        ("003", None, "invalid", "# Unfinished RADIUS [0]\nradius = 3\n", None),
        ("004", 0, "keep", "radius = 4\n", "零分方案"),
        ("1000", -1, "discard", "x = 1000\n", "Repeat the earlier score"),
    ]
    ledger = []
    for attempt, score, decision, source, summary in records:
        directory = root / attempt
        directory.mkdir()
        (directory / ("draft.py" if attempt == "003" else "solver.py")).write_text(source)
        research.save(directory / "decision.json", {"decision": decision, "parent": "000"})
        result = {"valid": score is not None, "mean_score": score}
        if attempt == "003":
            result["error"] = "Agent did not submit within 128 model calls"
        if summary is not None:
            research.save(directory / "submission.json", {"summary": summary})
        ledger.append({"attempt": attempt, "decision": decision, **result})
    return agent.Workspace(root, "current = True\n", ledger)


def test_history_orders_scores_and_unsubmitted_drafts(tmp_path):
    workspace = history_workspace(tmp_path)
    assert [r["attempt"] for r in workspace.list_attempts()["attempts"]] == [
        "1000",
        "004",
        "003",
        "002",
        "001",
        "000",
    ]
    scores = workspace.list_attempts(order="score")["attempts"]
    assert [r["attempt"] for r in scores] == ["004", "1000", "001", "000", "003", "002"]
    page = workspace.list_attempts(order="score", offset=1, limit=2)
    assert page["total"] == 6 and [r["attempt"] for r in page["attempts"]] == ["1000", "001"]
    draft = workspace.read_attempt("003")
    assert draft["source"] == "# Unfinished RADIUS [0]\nradius = 3\n"
    assert draft["source_kind"] == "draft" and draft["status"] == "unsubmitted"
    assert draft["summary_source"] == "system"
    assert "128 model calls" in draft["summary"] and "not evaluated" in draft["summary"]
    submitted = workspace.read_attempt("002")
    assert submitted["status"] == "submitted" and not submitted["valid"]
    assert submitted["source_kind"] == "solver" and submitted["summary_source"] == "model"
    (tmp_path / "001/submission.json").unlink()
    legacy = agent.attempt_info(tmp_path / "001", {"valid": True})
    assert legacy["status"] == "submitted" and legacy["summary"] is None


def test_history_search_is_literal_paginated_and_includes_unevaluated_code(tmp_path):
    workspace = history_workspace(tmp_path)
    matches = workspace.search_attempts("RaDiUs", field="code", order="score", limit=2)
    assert matches["total"] == 3
    assert [r["attempt"] for r in matches["attempts"]] == ["004", "001"]
    assert (
        workspace.search_attempts("radius", field="code", order="score", offset=2)["attempts"][0][
            "attempt"
        ]
        == "003"
    )
    summaries = workspace.search_attempts("radius", field="summary")["attempts"]
    assert [r["attempt"] for r in summaries] == ["001"]
    assert summaries[0]["matches"][0]["field"] == "summary"
    assert workspace.search_attempts("零分", field="summary")["total"] == 1
    literal = workspace.search_attempts("[0]", field="code")
    assert literal["total"] == 1
    assert literal["attempts"][0]["matches"] == [
        {"field": "code", "line": 1, "text": "# Unfinished RADIUS [0]"}
    ]
    (tmp_path / ".env").write_text("SECRET_OUTSIDE_HISTORY")
    assert workspace.search_attempts("SECRET_OUTSIDE_HISTORY")["total"] == 0
    assert workspace.search_attempts("missing")["attempts"] == []
    assert workspace.search_attempts("radius", offset=99)["attempts"] == []


@pytest.mark.parametrize(
    "selector,expected",
    [
        ("000", "000"),
        ("001", "001"),
        ("002", "002"),
        ("003", "003"),
        ("latest", "1000"),
        ("latest_valid", "1000"),
        ("best", "004"),
    ],
)
def test_load_solver_selects_history_then_allows_edits_without_changing_saved_versions(
    tmp_path, selector, expected
):
    workspace = history_workspace(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    history = workspace.list_attempts()
    original = workspace.read_attempt(expected)
    result = workspace.dispatch(call("load_solver", {"attempt": selector}))
    assert result == {"loaded": True, **original}
    assert workspace.source == original["source"]
    assert not workspace.submitted and workspace.summary is None
    last_line = workspace.source.splitlines()[-1]
    patch = f"@@\n {last_line}\n+# Edited after loading\n*** End of File"
    assert workspace.dispatch(call("edit_solver", {"patch": patch})) == {"edited": True}
    assert workspace.source == original["source"] + "# Edited after loading\n"
    assert workspace.dispatch(call("submit", {"summary": "Refine an earlier version."}))[
        "submitted"
    ]
    frozen = workspace.source
    assert (
        "already submitted" in workspace.dispatch(call("load_solver", {"attempt": "000"}))["error"]
    )
    assert workspace.source == frozen
    assert workspace.list_attempts() == history
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_latest_can_load_an_unsubmitted_draft_while_latest_valid_skips_failures(tmp_path):
    workspace = history_workspace(tmp_path)
    workspace.ledger = {k: v for k, v in workspace.ledger.items() if int(k) <= 3}
    loaded = workspace.load_solver("latest")
    assert loaded["attempt"] == "003" and loaded["source_kind"] == "draft"
    assert loaded["status"] == "unsubmitted" and "not evaluated" in loaded["summary"]
    assert workspace.load_solver("latest_valid")["attempt"] == "001"
    assert workspace.load_solver("best")["attempt"] == "001"


@pytest.mark.parametrize("selector", ["latest", "latest_valid", "best"])
def test_load_shortcuts_include_the_baseline_and_explain_empty_history(tmp_path, selector):
    empty = agent.Workspace(tmp_path, "current = True\n", [])
    error = empty.dispatch(call("load_solver", {"attempt": selector}))["error"]
    assert f"No previous attempt matches {selector!r}" in error and "list_attempts" in error
    assert empty.source == "current = True\n" and not empty.submitted
    workspace = history_workspace(tmp_path)
    workspace.ledger = {"000": workspace.ledger["000"]}
    assert workspace.load_solver(selector)["attempt"] == "000"


@pytest.mark.parametrize("score", [0, 5e-10])
def test_load_best_keeps_the_retained_version_when_a_newer_score_does_not_clear_threshold(
    tmp_path, score
):
    workspace = history_workspace(tmp_path)
    workspace.ledger["1000"]["mean_score"] = score
    assert workspace.load_solver("best")["attempt"] == "004"
    assert workspace.load_solver("latest_valid")["attempt"] == "1000"


def test_loading_missing_source_fails_without_changing_the_draft(tmp_path):
    history = history_workspace(tmp_path)
    (tmp_path / "003/draft.py").unlink()
    workspace = agent.Workspace(tmp_path, "current = True\n", history.ledger.values())
    result = workspace.dispatch(call("load_solver", {"attempt": "003"}))
    assert "Attempt '003' has no saved solver or draft" in result["error"]
    assert "source_kind solver or draft" in result["error"]
    assert workspace.source == "current = True\n" and not workspace.submitted


def test_loading_a_draft_defers_syntax_checks_until_submission(tmp_path):
    workspace = history_workspace(tmp_path)
    (tmp_path / "003/draft.py").write_text("def unfinished(")
    assert workspace.load_solver("003")["source"] == "def unfinished("
    result = workspace.dispatch(call("submit", {"summary": "Continue the draft."}))
    assert "SyntaxError" in result["error"] and not workspace.submitted


def test_history_search_bounds_snippets_around_the_match(tmp_path):
    workspace = history_workspace(tmp_path)
    line = "# " + "a" * 900 + "NEEDLE" + "b" * 900
    (tmp_path / "003/draft.py").write_text("\n".join([line] * 8))
    snippets = workspace.search_attempts("needle")["attempts"][0]["matches"]
    assert len(snippets) == 3
    assert all("NEEDLE" in m["text"] and len(m["text"]) <= 500 for m in snippets)


def test_unsubmitted_notes_are_bounded_and_do_not_require_a_model_summary(tmp_path):
    (tmp_path / "draft.py").write_text("x = 1\n")
    info = agent.attempt_info(tmp_path, {"error": "失败" * 500})
    assert len(info["summary"]) == agent.SUMMARY_LIMIT
    assert info["summary_source"] == "system" and info["status"] == "unsubmitted"
    assert info["source_kind"] == "draft"


def test_context_patch_applies_multiple_edits_and_preserves_indentation(tmp_path):
    source = "import math\n\ndef solve(n, seed, budget):\n    radius = 1\n    return radius\n"
    patch = "@@\n-import math\n+import random\n@@ def solve(n, seed, budget):\n-    radius = 1\n+    radius = 2\n+    # keep indentation\n     return radius\n"
    workspace = agent.Workspace(tmp_path, source, [])
    assert workspace.dispatch(call("edit_solver", {"patch": patch})) == {"edited": True}
    assert (
        workspace.source
        == "import random\n\ndef solve(n, seed, budget):\n    radius = 2\n    # keep indentation\n    return radius\n"
    )
    assert not workspace.submitted and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "source,patch,expected",
    [
        ("", "@@\n+x = 1", "x = 1\n"),
        ("x = 1\n", "@@\n-x = 1", ""),
        ("x = 1\n", "@@\n+import math\n x = 1", "import math\nx = 1\n"),
        ("x = 1\n", "@@\n x = 1\n+y = 2", "x = 1\ny = 2\n"),
        ("x = 1\n\ny = 2\n", "@@\n x = 1\n\n-y = 2\n+y = 3", "x = 1\n\ny = 3\n"),
        ("x = 1\nx = 1\n", "@@\n-x = 1\n+x = 2\n*** End of File", "x = 1\nx = 2\n"),
        (
            "def a():\n    return 1\ndef b():\n    return 1\n",
            "@@ def b():\n-    return 1\n+    return 2",
            "def a():\n    return 1\ndef b():\n    return 2\n",
        ),
        (
            "x = 1\n",
            "*** Begin Patch\n*** Update File: solver.py\n@@\n-x = 1\n+x = 2\n*** End Patch",
            "x = 2\n",
        ),
    ],
)
def test_context_patch_insert_delete_anchor_and_wrapper(source, patch, expected):
    assert agent.apply_solver_patch(source, patch) == expected


@pytest.mark.parametrize(
    "patch",
    [
        "",
        "replace everything",
        "@@\n x = 1",
        "@@\n+new = 1",
        "@@\n-missing\n+changed",
        "@@\n-x = 1\n+x = 2",
        "@@\n-missing\n+x = 2\n*** End of File",
        "*** Begin Patch\n*** Update File: ../outside.py\n@@\n+x\n*** End Patch",
        "*** Begin Patch\n*** Delete File: solver.py\n*** End Patch",
        "*** Begin Patch\n*** Update File: solver.py\n@@\n+x",
        "@@\n a = 0\n-x = 1\n+x = 2\n@@\n-missing\n+changed",
        "@@\n a = 0\n-x = 1\n+x = 2\n*** Update File: other.py\n@@\n+secret",
    ],
)
def test_bad_or_ambiguous_patches_are_atomic(tmp_path, patch):
    source = "a = 0\nx = 1\nx = 1\n"
    workspace = agent.Workspace(tmp_path, source, [])
    result = workspace.dispatch(call("edit_solver", {"patch": patch}))
    assert "error" in result and "expected_arguments" in result
    assert workspace.source == source and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize(
    "patch,hunk,line,detail",
    [
        ("@@\n-a = 0\n+a = 1\n@@\n-missing\n+changed", 2, 4, "context not found"),
        ("@@\n-x = 1\n+x = 2", 1, 1, "context matches 2 locations"),
        ("@@ absent\n-x = 1\n+x = 2", 1, 1, "anchor not found"),
        ("@@\nx = 1", 1, 2, "lines must start with a space, - or +"),
        ("@@\n x = 1", 1, 1, "each hunk must add or remove lines"),
    ],
)
def test_patch_errors_locate_the_failure_without_applying_earlier_hunks(
    tmp_path, wrapped, patch, hunk, line, detail
):
    source = "a = 0\nx = 1\nx = 1\n"
    if wrapped:
        patch = f"*** Begin Patch\n*** Update File: solver.py\n{patch}\n*** End Patch"
        line += 2
    workspace = agent.Workspace(tmp_path, source, [])
    result = workspace.dispatch(call("edit_solver", {"patch": patch}))
    assert f"Patch hunk {hunk}" in result["error"]
    assert f"patch line {line}" in result["error"]
    assert detail in result["error"]
    assert "This call did not change the draft or submit it" in result["error"]
    assert workspace.source == source and not workspace.submitted


def test_patch_accepts_unified_hunk_headers_without_relying_on_line_numbers():
    source = "\n".join(f"value_{i} = {i}" for i in range(30)) + "\n"
    replacement = (
        source.replace("value_3 = 3", "value_3 = 33").replace("value_25 = 25\n", "") + "# 结束\n"
    )
    diff = list(difflib.unified_diff(source.splitlines(True), replacement.splitlines(True)))
    assert agent.apply_solver_patch(source, "".join(diff[2:])) == replacement


def test_patch_preserves_unicode_separators_inside_source_strings():
    source = 'label = "甲\u2028乙"\nx = 1\n'
    assert agent.apply_solver_patch(source, "@@\n-x = 1\n+x = 2\n") == source.replace(
        "x = 1", "x = 2"
    )
    assert agent.apply_solver_patch("x = 1\r\n", "@@\r\n-x = 1\r\n+x = 2\r\n") == "x = 2\n"


@pytest.mark.parametrize("finish", ["error", None])
def test_incomplete_tool_calls_never_edit_or_submit(tmp_path, finish):
    workspace = agent.Workspace(tmp_path, "x = 1", [])
    error = agent.advance(
        response(
            [
                call("edit_solver", {"patch": "@@\n-x = 1\n+x = 2"}),
                call("submit", {"summary": "Change the parameter to two."}, "call-2"),
            ],
            finish,
        ),
        workspace,
        [],
    )
    assert "did not finish" in error
    assert not workspace.submitted and workspace.source == "x = 1"
    assert workspace.summary is None


def test_provider_error_is_included_in_the_round_failure(tmp_path):
    recorded = response([], "error")
    recorded["error"] = {"code": 502, "message": "Provider disconnected"}
    workspace = agent.Workspace(tmp_path, "x = 1\n", [])
    error = agent.advance(recorded, workspace, [])
    assert "502" in error and "Provider disconnected" in error
    assert not workspace.submitted


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"summary": ""},
        {"summary": " \n\t "},
        {"summary": "a" * 501},
        {"summary": "改" * 501},
        {"summary": None},
        {"summary": 42},
    ],
)
def test_submit_requires_a_short_nonblank_summary_and_allows_correction(tmp_path, args):
    workspace = agent.Workspace(tmp_path, "x = 1\n", [])
    assert "error" in workspace.dispatch(call("submit", args))
    assert not workspace.submitted and workspace.summary is None
    assert workspace.source == "x = 1\n"
    result = workspace.dispatch(call("submit", {"summary": "  Keep this implementation. \n"}))
    assert result == {"submitted": True, "summary": "Keep this implementation."}


def test_summary_limit_counts_unicode_characters(tmp_path):
    workspace = agent.Workspace(tmp_path, "x = 1\n", [])
    summary = "改" * 500
    assert workspace.dispatch(call("submit", {"summary": summary})) == {
        "submitted": True,
        "summary": summary,
    }


@pytest.mark.parametrize("task", TASKS)
def test_agent_prompt_keeps_task_contract_without_one_shot_directions(task):
    brief = Path(f"tasks/{task}/program.md").read_text()
    messages = agent.initial_messages(brief, "003", 1.25, 128, "007")
    assert [message["role"] for message in messages] == ["system", "user"]
    prompt = messages[1]["content"]
    assert "Python code block" not in prompt
    assert "Try one" not in prompt
    assert "Keep the planning brief" not in prompt
    assert "Current round: 007; 6 earlier research rounds completed" in prompt
    assert "including failures and unsubmitted rounds; baseline 000 excluded" in prompt
    assert "Current best: 003; score: 1.25" in prompt
    assert "up to 128 model calls" in prompt
    assert "Implement `solve(" in prompt
    assert "15 CPU seconds" in prompt
    for paragraph in brief.split("\n\n"):
        if any(marker in paragraph for marker in ("Implement `solve", "budget", "score", "RMSLE")):
            # The circle output paragraph also mentions the score but is not the task contract.
            if "Python code block" not in paragraph:
                assert paragraph in prompt


def test_default_budget_allows_submission_on_call_128(tmp_path, monkeypatch):
    args = agent.argument_parser().parse_args(["--run", str(tmp_path)])
    assert args.max_steps == 128
    assert args.max_tokens == 65536
    assert research.argument_parser().parse_args(["--run", str(tmp_path)]).max_tokens == 32768
    assert (
        agent.argument_parser()
        .parse_args(["--run", str(tmp_path), "--max-tokens", "8192"])
        .max_tokens
        == 8192
    )
    assert (
        agent.argument_parser().parse_args(["--run", str(tmp_path), "--max-steps", "256"]).max_steps
        == 256
    )
    best, directory = tmp_path / "000", tmp_path / "001"
    best.mkdir()
    directory.mkdir()
    source = "def solve(n, seed, budget):\n    return []\n"
    (best / "solver.py").write_text(source)
    research.save(best / "evaluation.json", {"mean_score": 0.0})
    manifest = {
        "agent": {"max_steps": args.max_steps},
        "task": "circle",
        "seeds": [0],
        "model": "test-model",
        "inference": {"endpoint": "https://api.example.com/v1"},
        "sampling": {"max_tokens": 1024},
    }
    calls = []

    def complete(body, endpoint, key, call_dir):
        calls.append(call_dir.name)
        assert len(body["messages"]) == 2 + 2 * (len(calls) - 1)
        return response(
            [
                call("submit", {"summary": "Keep the current solver for comparison."})
                if len(calls) == 128
                else call("read_solver")
            ]
        )

    monkeypatch.setattr(inference, "complete", complete)
    evaluated = []
    monkeypatch.setattr(
        agent, "evaluate", lambda source, *a: evaluated.append(source) or {"valid": True}
    )
    assert agent.propose(directory, manifest, best, [], None) == {"valid": True}
    assert len(calls) == 128 and calls[-1] == "128"
    assert evaluated == [source]


@pytest.mark.parametrize("corruption", ["json", "missing", "duplicate_id"])
def test_bad_tool_calls_are_recorded_without_execution(tmp_path, corruption):
    item = call("edit_solver", {"patch": "@@\n-x = 1\n+x = 2"})
    if corruption == "json":
        item["function"]["arguments"] = "{"
    elif corruption == "missing":
        item.pop("function")
    workspace = agent.Workspace(tmp_path, "x = 1", [])
    calls = [item, item] if corruption == "duplicate_id" else [item]
    messages = []
    error = agent.advance(response(calls), workspace, messages)
    assert error is None
    assert messages[-1]["role"] == "user"
    assert "None of its tools ran" in messages[-1]["content"]
    assert workspace.source == "x = 1" and not workspace.submitted


def arguments(root, max_steps=4):
    return SimpleNamespace(
        run=root,
        task="circle",
        experiments=1,
        resume=False,
        max_tokens=1024,
        max_steps=max_steps,
        base_url="https://api.example.com/v1",
        model="test-tool-model",
        api_key_env="AGENT_TEST_KEY",
    )


def baseline_evaluation(source):
    """Use a known result in tests that exercise orchestration, not candidate execution."""
    from tasks.circle import check
    from tasks.circle.baseline import solve

    checked = check(solve(TASKS["circle"].PROBLEM, 0, 15))
    return {
        "valid": True,
        "mean_score": checked["score"],
        "source_sha256": digest(source),
        "trials": [{"valid": True, "seed": 0, **checked}],
    }


def test_next_round_discovers_failed_draft_and_replays_patch_without_chat_history(
    tmp_path, monkeypatch
):
    baseline = Path("tasks/circle/baseline.py").read_text()
    draft = baseline + "\n# HISTORY_MARKER: unfinished search idea\n"
    last_line = draft.splitlines()[-1]
    patch = f"@@\n {last_line}\n+# Revisit HISTORY_MARKER\n*** End of File"
    plans = [
        [call("edit_solver", {"source": draft})],
        [],  # Stops without submit, preserving an unevaluated draft.
        [
            call("list_attempts", {"order": "score"}, "history"),
            call("search_attempts", {"query": "history_marker", "field": "code"}, "search"),
            call("read_attempt", {"attempt": "001"}, "draft"),
            call("load_solver", {"attempt": "latest"}, "load"),
        ],
        [
            call("edit_solver", {"patch": patch}),
            call("submit", {"summary": "Revisit the earlier draft's idea."}, "submit"),
        ],
    ]
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        calls = plans[len(requests) - 1]
        return stream(calls, "tool_calls" if calls else "stop")

    def evaluate(source, seeds, task):
        assert source in (baseline, draft + "# Revisit HISTORY_MARKER\n")
        return baseline_evaluation(source)

    monkeypatch.setenv("AGENT_TEST_KEY", "test-credential")
    monkeypatch.setattr(inference.OPENER, "open", respond)
    monkeypatch.setattr(research, "evaluate", evaluate)
    monkeypatch.setattr(agent, "evaluate", evaluate)
    args = arguments(tmp_path / "run", max_steps=2)
    args.experiments = 2
    research.run(args, agent=agent.propose)
    assert len(requests) == 4
    fresh = requests[2]["messages"]
    assert [m["role"] for m in fresh] == ["system", "user"]
    assert "1 earlier research rounds completed" in fresh[1]["content"]
    assert "Current best: 000; score:" in fresh[1]["content"]
    assert "HISTORY_MARKER" not in json.dumps(fresh)
    feedback = {
        m["tool_call_id"]: json.loads(m["content"])
        for m in requests[3]["messages"]
        if m["role"] == "tool"
    }
    assert [r["attempt"] for r in feedback["history"]["attempts"]] == ["000", "001"]
    assert feedback["search"]["attempts"][0]["status"] == "unsubmitted"
    assert feedback["draft"]["source"] == draft
    assert feedback["load"]["loaded"] and feedback["load"]["attempt"] == "001"
    assert (args.run / "002/solver.py").read_text() == draft + "# Revisit HISTORY_MARKER\n"
    assert json.loads((args.run / "002/decision.json").read_text())["parent"] == "000"
    target = tmp_path / "export"
    export(args.run, target)
    verify(target)
    assert (target / "001/draft.py").read_text() == draft
    manifest = json.loads((target / "manifest.json").read_text())
    (target / "001/draft.py").write_text("# Unrecorded draft change\n")
    with pytest.raises(AssertionError):
        verify_agent(target / "002", manifest)


@pytest.mark.parametrize("failure", ["json", "length", "duplicate_id", "missing_id", "array"])
def test_recovery_preserves_evidence_and_never_replays_or_executes_bad_calls(
    tmp_path, monkeypatch, failure, capsys
):
    baseline = Path("tasks/circle/baseline.py").read_text()
    edit = call("edit_solver", {"source": "x = 2\n"})
    submit = call("submit", {"summary": "Keep the draft."}, "submit")
    finish = "tool_calls"
    if failure == "json":
        submit["function"]["arguments"] = '{"summary": "cut'
    elif failure == "length":
        finish = "length"
    elif failure == "duplicate_id":
        submit["id"] = edit["id"]
    elif failure == "missing_id":
        submit["id"] = ""
    else:
        submit["function"]["arguments"] = "[]"
    plans = [
        ([edit, submit], finish),
        ([call("submit", {"new": "Keep the unchanged solver."})], "tool_calls"),
        ([call("submit", {"summary": "改" * 501})], "tool_calls"),
        ([call("submit", {"summary": "Keep the unchanged solver."})], "tool_calls"),
    ]
    requests = []

    def respond(request, **kwargs):
        requests.append(json.loads(request.data))
        return stream(*plans[len(requests) - 1])

    def evaluate(source, seeds, task):
        assert source == baseline  # Even the valid edit beside a malformed call must not run.
        return baseline_evaluation(source)

    monkeypatch.setenv("AGENT_TEST_KEY", "test-credential")
    monkeypatch.setattr(inference.OPENER, "open", respond)
    monkeypatch.setattr(research, "evaluate", evaluate)
    monkeypatch.setattr(agent, "evaluate", evaluate)
    args = arguments(tmp_path / "run", max_steps=4)
    research.run(args, agent=agent.propose)
    assert len(requests) == 4
    assert [m["role"] for m in requests[1]["messages"]] == ["system", "user", "user"]
    assert "None of its tools ran" in requests[1]["messages"][-1]["content"]
    correction = json.loads(requests[2]["messages"][-1]["content"])
    assert "missing required arguments: summary" in correction["error"]
    assert correction["expected_arguments"]["required"] == ["summary"]
    length_error = json.loads(requests[3]["messages"][-1]["content"])["error"]
    assert "received 501 characters (maximum 500; 1 over)" in length_error
    output = capsys.readouterr().out
    assert correction["error"] in output and length_error in output
    recorded = json.loads((args.run / "001/calls/001/response.json").read_text())
    assert recorded["tool_calls"] == [edit, submit]
    assert recorded["finish_reason"] == finish
    assert (args.run / "001/solver.py").read_text() == baseline
    target = tmp_path / "export"
    export(args.run, target)
    verify(target)
    manifest = json.loads((target / "manifest.json").read_text())
    manifest["fixed_files"]["research_agent.py"] = "a different agent implementation"
    with pytest.raises(AssertionError, match="original source version"):
        verify_agent(target / "001", manifest)


def test_agent_run_resume_export_report_and_replay(tmp_path, monkeypatch):
    improved = """def solve(n, seed, budget):
    circles = [[(x + 0.5) / 5, (y + 0.5) / 5, 0.099]
               for x in range(5) for y in range(5)]
    return circles + [[0.2, 0.2, 0.04]]
"""
    plans = [
        [call("list_attempts", {"limit": 1}), call("read_solver", identifier="read")],
        [call("read_attempt", {"attempt": "000"})],
        [
            call("edit_solver", {"source": improved}),
            call(
                "submit",
                {"summary": "Use 25 large circles and a gap circle to increase total radius."},
                "submit",
            ),
        ],
        [call("list_attempts"), call("read_solver", identifier="read")],
        [call("read_attempt", {"attempt": "001"})],
        [call("submit", {"summary": "Recheck the unchanged <solver> & compare the score."})],
    ]
    requests = []
    token = "private-agent-test-credential"
    monkeypatch.setenv("AGENT_TEST_KEY", token)

    def respond(request, **kwargs):
        assert request.get_header("Authorization") == "Bearer " + token
        requests.append(json.loads(request.data))
        return stream(plans[len(requests) - 1])

    monkeypatch.setattr(inference.OPENER, "open", respond)
    args = arguments(tmp_path / "run")
    research.run(args, agent=agent.propose)
    args.resume, args.experiments, args.max_steps = True, 2, 99
    research.run(args, agent=agent.propose)
    manifest = json.loads((args.run / "manifest.json").read_text())
    assert manifest["agent"] == {"max_steps": 4}
    assert "research_agent.py" in manifest["fixed_files"]
    assert [m["role"] for m in requests[0]["messages"]] == ["system", "user"]
    assert [m["role"] for m in requests[3]["messages"]] == ["system", "user"]
    assert "Current best: 001" in requests[3]["messages"][1]["content"]
    assert requests[1]["messages"][2]["reasoning"] == "Plan an edit."
    assert "reasoning_details" not in requests[1]["messages"][2]
    assert json.loads(requests[4]["messages"][-1]["content"])["source"] == improved
    history = json.loads(requests[4]["messages"][-2]["content"])["attempts"]
    assert (
        history[0]["summary"] == "Use 25 large circles and a gap circle to increase total radius."
    )
    assert history[1]["summary"] == "Baseline solver."
    assert history[1]["summary_source"] == "system"
    assert history[0]["summary_source"] == "model"
    assert json.loads(requests[5]["messages"][-1]["content"])["summary"] == history[0]["summary"]
    assert json.loads((args.run / "001/decision.json").read_text()) == {
        "decision": "keep",
        "parent": "000",
    }
    assert json.loads((args.run / "002/decision.json").read_text()) == {
        "decision": "discard",
        "parent": "001",
    }
    with pytest.raises(RuntimeError, match="mode changed"):
        research.run(args)
    target = tmp_path / "export"
    export(args.run, target)
    verify(target)
    assert inventory(target)["distinct_scored_candidates"] == 1
    assert "Agent conversation and tool results" in (target / "report.html").read_text()
    html = (target / "report.html").read_text()
    assert history[0]["summary"] in html
    assert "Recheck the unchanged &lt;solver&gt; &amp; compare the score." in html
    assert "<details><summary>Agent conversation and tool results" in html
    assert "<th>Attempt summary</th>" in html
    assert "&quot;reasoning_details&quot;" not in html
    assert "Plan an edit." in html  # Reasoning remains in its dedicated section and raw transcript.
    assert (
        json.loads((target / "001/submission.json").read_text())["summary"] == history[0]["summary"]
    )
    assert "001/calls/001/stream.jsonl.gz" in (target / "report.html").read_text()
    assert all(token not in p.read_text() for p in args.run.rglob("*") if p.is_file())
    for step in (target / "001/calls").iterdir():
        assert (step / "stream.jsonl.gz").is_file()
    submission = target / "001/submission.json"
    original = submission.read_text()
    research.save(submission, {"summary": "An unrecorded explanation."})
    with pytest.raises(AssertionError):
        verify_agent(target / "001", manifest)
    submission.write_text(original)
    changed = target / "001/solver.py"
    changed.write_text(improved + "# unrecorded edit\n")
    with pytest.raises(AssertionError):
        verify_agent(target / "001", manifest)
    changed.write_text(improved)
    request_file = target / "002/calls/001/request.json"
    request = json.loads(request_file.read_text())
    request["messages"][1]["content"] += "\nIncorrect prior attempt count or task."
    research.save(request_file, request)
    with pytest.raises(AssertionError):
        verify_agent(target / "002", manifest)
    request["messages"][1]["content"] = requests[3]["messages"][1]["content"]
    request["messages"].append({"role": "assistant", "content": "Previous round"})
    research.save(request_file, request)
    with pytest.raises(AssertionError):
        verify_agent(target / "002", manifest)


@pytest.mark.parametrize(
    "failure", ["no_submit", "step_limit", "truncated", "bad_solver", "bad_evaluation"]
)
def test_failed_rounds_are_saved_and_never_silently_submitted(tmp_path, monkeypatch, failure):
    baseline = Path("tasks/circle/baseline.py").read_text()
    if failure == "no_submit":
        calls, finish = [], "stop"
    elif failure == "step_limit":
        calls, finish = [call("read_solver")], "tool_calls"
    elif failure == "truncated":
        calls, finish = [call("submit", {"summary": "Keep the current solver."})], "length"
    else:
        calls, finish = (
            [
                call(
                    "edit_solver",
                    {
                        "source": "def solve(:"
                        if failure == "bad_solver"
                        else "def solve(n, seed, budget):\n    return []\n",
                    },
                ),
                call(
                    "submit",
                    {"summary": "Replace the solver to test a different output."},
                    "submit",
                ),
            ],
            "tool_calls",
        )
    monkeypatch.setenv("AGENT_TEST_KEY", "test-credential")
    monkeypatch.setattr(inference.OPENER, "open", lambda *a, **kw: stream(calls, finish))

    def evaluate(source, seeds, task):
        assert source == baseline  # Only the baseline may run in this test.
        return baseline_evaluation(source)

    monkeypatch.setattr(research, "evaluate", evaluate)
    monkeypatch.setattr(
        agent,
        "evaluate",
        lambda source, *a: {
            "valid": False,
            "mean_score": None,
            "source_sha256": digest(source),
            "error": "Simulated invalid solution",
        }
        if failure == "bad_evaluation"
        else pytest.fail("Unsubmitted code executed"),
    )
    args = arguments(tmp_path / "run", max_steps=1)
    research.run(args, agent=agent.propose)
    assert (args.run / "001/solver.py").exists() == (failure == "bad_evaluation")
    assert (args.run / "001/submission.json").exists() == (failure == "bad_evaluation")
    assert (args.run / "001/draft.py").exists()
    result = json.loads((args.run / "001/evaluation.json").read_text())
    assert not result["valid"] and result["error"]
    target = tmp_path / "export"
    export(args.run, target)
    verify(target)
    ledger = [{"attempt": "001", "decision": "invalid", **research.feedback(result)}]
    history = agent.Workspace(target, baseline, ledger)
    record = history.read_attempt("001")
    html = (target / "report.html").read_text()
    if failure == "bad_evaluation":
        assert record["status"] == "submitted" and record["summary_source"] == "model"
        assert record["source_kind"] == "solver"
    else:
        assert record["status"] == "unsubmitted" and record["summary_source"] == "system"
        assert record["source_kind"] == "draft"
        assert "Not submitted; draft not evaluated." in record["summary"]
        assert result["error"] in record["summary"]
        assert record["source"] == (target / "001/draft.py").read_text()
        assert "System note:" in html and 'href="001/draft.py"' in html


def test_agent_step_limit_is_checked_before_connecting(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "model_server", lambda *a: pytest.fail("Connected"))
    with pytest.raises(ValueError, match="positive"):
        research.run(arguments(tmp_path / "run", max_steps=0), agent=agent.propose)


@pytest.mark.parametrize(
    "failed_call",
    [
        call("edit_solver", {"patch": "@@\n-missing\n+x = 3"}, "bad-edit"),
        call("load_solver", {"attempt": "999"}, "bad-load"),
    ],
)
def test_failed_mutation_blocks_same_response_submission(tmp_path, failed_call):
    workspace = agent.Workspace(tmp_path, "x = 1\n", [])
    messages = []
    agent.advance(
        response(
            [
                call("edit_solver", {"source": "x = 2\n"}, "good-edit"),
                failed_call,
                call("submit", {"summary": "Changed x to three."}, "submit"),
            ]
        ),
        workspace,
        messages,
    )
    assert workspace.source == "x = 2\n" and not workspace.submitted
    assert "skipped" in json.loads(messages[-1]["content"])["error"]
    assert len([m for m in messages if m["role"] == "tool"]) == 3
    agent.advance(response([call("submit", {"summary": "Keep x at two."})]), workspace, messages)
    assert workspace.submitted and workspace.summary == "Keep x at two."


@pytest.mark.parametrize("path", ["solver.py", "../../outside.py", None, {"anything": True}])
def test_edit_ignores_path_with_feedback_and_never_uses_it(tmp_path, path):
    workspace = agent.Workspace(tmp_path, "x = 1\n", [])
    result = workspace.dispatch(call("edit_solver", {"path": path, "source": "x = 2\n"}))
    assert result["edited"] and "Ignored path" in result["notice"]
    assert workspace.source == "x = 2\n" and not list(tmp_path.iterdir())
    error = workspace.dispatch(call("edit_solver", {"path": path, "patch": "bad patch"}))
    assert "error" in error and "Ignored path" in error["notice"]
    assert workspace.source == "x = 2\n"
    assert workspace.dispatch(call("edit_solver", {"source": "x = 2\n"})) == {"edited": True}


def test_search_matches_terms_across_fields_and_lines_without_query_advice(tmp_path):
    workspace = history_workspace(tmp_path)
    matches = workspace.search_attempts("Unfinished 3", mode="all")
    assert [r["attempt"] for r in matches["attempts"]] == ["003"]
    matches = workspace.search_attempts("Tune 1", mode="all")
    assert [r["attempt"] for r in matches["attempts"]] == ["001"]
    assert workspace.search_attempts("Tune 1", field="code")["total"] == 0
    assert workspace.search_attempts("Tune 1", mode="literal") == {
        "query": "Tune 1",
        "mode": "literal",
        "total": 0,
        "attempts": [],
    }
    assert workspace.search_attempts("Tune missing", mode="any")["total"] == 1
    assert workspace.search_attempts("Tune the", mode="literal")["total"] == 1
    assert workspace.search_attempts("[0]", mode="literal")["total"] == 1
    assert "error" in workspace.dispatch(call("search_attempts", {"query": "x", "mode": "regex"}))


def test_line_ranges_read_current_and_historical_source_without_mutation(tmp_path):
    workspace = history_workspace(tmp_path)
    workspace.source = "first\nsecond\nthird"
    assert workspace.read_solver(start_line=2, end_line=2) == {
        "source": "second\n",
        "start_line": 2,
        "end_line": 2,
        "total_lines": 3,
    }
    assert workspace.read_solver(start_line=2, end_line=200)["source"] == "second\nthird"
    assert workspace.read_solver(end_line=1)["source"] == "first\n"
    assert workspace.read_solver() == {"source": "first\nsecond\nthird"}
    record = workspace.read_attempt("003", start_line=2)
    assert record["source"] == "radius = 3\n" and record["total_lines"] == 2
    assert record["source_kind"] == "draft" and workspace.source == "first\nsecond\nthird"
    assert workspace.load_solver("003")["source"] == "# Unfinished RADIUS [0]\nradius = 3\n"
    for args in (
        {"start_line": 0},
        {"start_line": 99},
        {"start_line": 2, "end_line": 1},
        {"start_line": True},
        {"end_line": "2"},
    ):
        assert "error" in workspace.dispatch(call("read_solver", args))
    workspace.source = ""
    assert "empty (0 lines)" in workspace.dispatch(call("read_solver", {"start_line": 1}))["error"]


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\v", "\f"])
def test_read_and_search_line_numbers_follow_python_lines(tmp_path, separator):
    workspace = history_workspace(tmp_path)
    source = f'label = "a{separator}b"\ntarget = 2\n'
    compile(source, "solver.py", "exec")
    workspace.source = source
    (tmp_path / "001/solver.py").write_text(source)
    expected = {"source": "target = 2\n", "start_line": 2, "end_line": 2, "total_lines": 2}
    assert workspace.read_solver(2, 2) == expected
    record = workspace.read_attempt("001", 2, 2)
    assert all(record[k] == v for k, v in expected.items())
    found = workspace.search_attempts("target", field="code")["attempts"]
    assert len(found) == 1
    assert found[0]["matches"] == [{"field": "code", "line": 2, "text": "target = 2"}]
    workspace.source = source.replace("\n", "\r\n")
    assert workspace.read_solver(2, 2) == {**expected, "source": "target = 2\r\n"}


@pytest.mark.parametrize("task", [name for name in TASKS if name != "circle"])
def test_agent_submission_export_and_replay_for_other_tasks(tmp_path, monkeypatch, task):
    source = Path(f"tasks/{task}/baseline.py").read_text() + "\n# Recheck the baseline.\n"
    calls = [
        call("load_solver", {"attempt": "best"}, "load"),
        call("edit_solver", {"source": source}, "edit"),
        call("submit", {"summary": "Recheck the task baseline."}, "submit"),
    ]
    monkeypatch.setenv("AGENT_TEST_KEY", "test-credential")
    monkeypatch.setattr(inference.OPENER, "open", lambda *a, **kw: stream(calls))
    args = arguments(tmp_path / "run", max_steps=1)
    args.task = task
    research.run(args, agent=agent.propose)
    result = json.loads((args.run / "001/evaluation.json").read_text())
    assert result["valid"]
    assert json.loads((args.run / "manifest.json").read_text())["task"] == task
    assert result["source_sha256"] == digest(source)
    target = tmp_path / "export"
    export(args.run, target)
    verify(target)
    assert "Recheck the task baseline." in (target / "report.html").read_text()


def test_report_and_verification_order_attempt_ids_numerically(tmp_path, monkeypatch):
    from tasks.circle import check
    from tasks.circle.baseline import solve

    root = tmp_path / "run"
    root.mkdir()
    research.save(
        root / "manifest.json",
        {"task": "circle", "model": "test-model", "seeds": [0], "inference": {"api_key_env": None}},
    )
    baseline = solve(26, 0, 15)
    parent = None
    # A small recorded fixture spanning the three/four-digit boundary.
    for number, scale in (("000", 0.5), ("999", 0.75), ("1000", 1.0)):
        directory = root / number
        directory.mkdir()
        circles = [[x, y, r * scale] for x, y, r in baseline]
        source = f"def solve(n, seed, budget):\n    return {circles!r}\n"
        checked = check(circles)
        research.save(
            directory / "evaluation.json",
            {
                "valid": True,
                "mean_score": checked["score"],
                "source_sha256": digest(source),
                "trials": [{"valid": True, "seed": 0, **checked}],
            },
        )
        (directory / "solver.py").write_text(source)
        research.save(
            directory / "decision.json",
            {"decision": "baseline" if parent is None else "keep", "parent": parent},
        )
        if parent is not None:
            chunk = {
                "choices": [
                    {"delta": {"content": f"```python\n{source}```"}, "finish_reason": "stop"}
                ]
            }
            data = f"data: {json.dumps(chunk)}\n".encode()
            monkeypatch.setattr(inference.OPENER, "open", lambda *a, **kw: io.BytesIO(data))
            inference.complete({}, "https://example.com/v1", None, directory)
        parent = number
    research.save(
        root / "summary.json",
        {"best_attempt": "1000", "best_score": checked["score"], "completed_experiments": 2},
    )
    (root / "best.py").write_text(source)
    target = tmp_path / "export"
    export(root, target)
    verify(target)
    assert [row["attempt"] for row in inventory(target)["records"]] == ["999", "1000"]
    document = (target / "report.html").read_text()
    assert document.index('id="attempt-999"') < document.index('id="attempt-1000"')


def test_history_exposes_generic_failure_causes_separately_from_model_summary(tmp_path):
    old = history_workspace(tmp_path)
    rows = list(old.ledger.values())
    rows[2].update(
        trials=[{"valid": False, "failure_kind": "timeout", "error": "CPU time limit exceeded"}]
    )
    rows[3].update(failure_kind="step_limit")
    workspace = agent.Workspace(tmp_path, "x = 1\n", rows)
    records = {r["attempt"]: r for r in workspace.list_attempts()["attempts"]}
    assert records["002"]["failure_kind"] == "timeout"
    assert records["002"]["error_summary"] == "CPU time limit exceeded"
    assert records["002"]["summary"] == "Try an optimizer"
    assert records["003"]["failure_kind"] == "step_limit"
    assert "128 model calls" in records["003"]["error_summary"]
    assert records["001"]["failure_kind"] is None and records["001"]["error_summary"] is None


def test_eof_patch_feedback_identifies_existing_context_without_relaxing_anchor(tmp_path):
    workspace = agent.Workspace(tmp_path, "x = 1\ny = 2\n", [])
    result = workspace.dispatch(
        call("edit_solver", {"patch": "@@\n-x = 1\n+x = 3\n*** End of File"})
    )
    assert "requires the hunk to end at EOF" in result["error"]
    assert "line(s) [1]" in result["error"]
    assert workspace.source == "x = 1\ny = 2\n"
