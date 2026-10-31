"""Fresh conversations that inspect past attempts, edit a solver and submit once."""

import io
import json
import re

import research
from runner import inference
from runner.core import failure_info, save
from runner.evaluate import ROOT, evaluate

SUMMARY_LIMIT = 500
HISTORY_FIELDS = (
    "attempt",
    "summary",
    "summary_source",
    "status",
    "source_kind",
    "decision",
    "valid",
    "mean_score",
    "failure_kind",
    "error_summary",
)
HISTORY_OPTIONS = {
    "order": {
        "type": "string",
        "enum": ["recent", "score"],
        "default": "recent",
        "description": "Newest first, or highest valid score first; ties use newest first, "
        "and unscored/invalid attempts come last.",
    },
    "offset": {"type": "integer", "minimum": 0, "default": 0},
    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 20},
}
LINE_OPTIONS = {
    "start_line": {"type": "integer", "minimum": 1, "description": "First line, 1-based."},
    "end_line": {
        "type": "integer",
        "minimum": 1,
        "description": "Last line, inclusive; defaults to EOF.",
    },
}


def tool(name, description, properties=None, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    tool(
        "list_attempts",
        "List completed earlier attempts, including the baseline and failures. "
        "Measured failure_kind and error_summary are separate from submission summaries. "
        "Unsubmitted attempts include a system explanation and the last unevaluated draft. "
        "Choose recent or score order and paginate with offset and limit.",
        HISTORY_OPTIONS,
    ),
    tool(
        "search_attempts",
        "Search earlier summaries and/or code, case-insensitively. mode=all (default) requires "
        "every whitespace-separated term somewhere in the selected fields of an attempt; "
        "mode=any requires at least one term; mode=literal matches the entire query as a "
        "continuous phrase. Terms are literal substrings, not regexes. Includes unsubmitted drafts. "
        "Return the mode, query and matching attempts with up to three line snippets each; "
        "zero total means no matches. read_attempt provides the complete source.",
        {
            "query": {
                "type": "string",
                "description": "A nonblank, single-line keyword or phrase.",
            },
            "field": {"type": "string", "enum": ["all", "summary", "code"], "default": "all"},
            "mode": {"type": "string", "enum": ["all", "any", "literal"], "default": "all"},
            **HISTORY_OPTIONS,
        },
        ("query",),
    ),
    tool(
        "read_attempt",
        "Read one previous attempt's summary, source, evaluation feedback and decision. "
        "For an unsubmitted attempt, source is its last draft, marked as unevaluated. "
        "Optionally select inclusive 1-based start_line/end_line; omit both for the full source.",
        {
            "attempt": {"type": "string", "description": "Attempt ID, for example 003."},
            **LINE_OPTIONS,
        },
        ("attempt",),
    ),
    tool(
        "load_solver",
        "Replace the current draft with an earlier attempt's solver or last unsubmitted draft. "
        "Use an attempt ID, latest (most recent round, including unsubmitted drafts), "
        "latest_valid (most recent valid evaluation, even if discarded), or best "
        "(current retained best). Baseline 000 is included. Return the resolved attempt's "
        "metadata and loaded source. Replaces any current edits; does not execute or submit code.",
        {
            "attempt": {
                "type": "string",
                "description": "Attempt ID such as 003, or latest, latest_valid, best.",
            }
        },
        ("attempt",),
    ),
    tool(
        "read_solver",
        "Read the current editable solver.py, initially copied from the best. "
        "Optionally select inclusive 1-based start_line/end_line; omit both for the full source. "
        "Range reads include actual start/end lines and total_lines; end_line is capped at EOF.",
        LINE_OPTIONS,
    ),
    tool(
        "edit_solver",
        "There is only one editable file, solver.py; no path argument is needed. "
        "A supplied path is ignored and reported in the result. Provide exactly one of patch or source. "
        "patch uses Codex-style context hunks: @@, then lines prefixed with a space "
        "(unchanged), - (remove), or + (add). Include exact, unique context and preserve "
        "indentation; line numbers are unnecessary. Multiple hunks apply atomically in file "
        "order. An optional '@@ exact source line' locates the following hunk after that line; "
        "'*** End of File' anchors a hunk to the file end. Example: "
        "'@@\n def solve(n, seed, budget):\n-    x = 1\n+    x = 2'. "
        "The optional *** Begin Patch / *** Update File: solver.py / *** End Patch wrapper "
        "is accepted. For a full rewrite, provide source with the complete new file instead. "
        "Edits are not evaluated; syntax is checked on submit.",
        {
            "patch": {"type": "string", "description": "Context diff for solver.py."},
            "source": {"type": "string", "description": "Complete replacement for solver.py."},
        },
    ),
    tool(
        "submit",
        "Submit the current solver.py for one sandboxed evaluation and end this round. "
        "Include a brief summary of what changed and why, for future history browsing. "
        "Invalid summaries and Python syntax errors leave the round open for correction.",
        {
            "summary": {
                "type": "string",
                "description": "A nonblank explanation of the change and its rationale, "
                "at most 500 Unicode characters including whitespace. No fixed format is required.",
                "minLength": 1,
                "maxLength": SUMMARY_LIMIT,
            }
        },
        ("summary",),
    ),
]

INSTRUCTIONS = """Develop a solver that improves the task's measured score while satisfying
its input/output contract and constraints. You choose the approach and how to use the
available tools. You may revise any part of the solver; there is no prescribed action sequence.

This is one round of an ongoing search. Earlier conversations are not included. Your
editable draft starts from the current best solver. Past code, submission summaries and
measured outcomes are available through the history tools, including unsuccessful attempts.
You can browse by recency or score and search summaries or code. Unsubmitted attempts have
system explanations and saved drafts; those drafts have not been evaluated.
Use load_solver(attempt) to start from a different version without copying its code into an edit;
it accepts an earlier attempt ID or the shortcuts latest, latest_valid and best.

Editing does not execute code. submit(summary) freezes the draft for one sandboxed evaluation
and ends the round. Provide a nonblank summary of what changed and why, at most 500 characters,
so later rounds can locate the work. The evaluator determines validity and score; a valid
score exceeding the current best by more than 1e-9 is retained. Higher scores are better.
The result is recorded for subsequent rounds. A plain-text answer does not submit a candidate.
Only the draft is editable; the tools provide no shell or access to other files.
There is only one editable file, solver.py. Do not pass path to edit_solver; if supplied,
it is ignored and reported, and the edit still targets solver.py.
"""


def task_brief(text):
    """Keep the task specification from program.md without its one-shot instructions."""
    boilerplate = ("Can you ", "Give a short explanation", "Try one idea at a time.")
    return "\n\n".join(
        paragraph
        for paragraph in text.strip().split("\n\n")
        if not paragraph.startswith(boilerplate)
    )


def initial_messages(brief, best, score, max_steps, attempt):
    return [
        {"role": "system", "content": INSTRUCTIONS},
        {
            "role": "user",
            "content": f"Task specification:\n{task_brief(brief)}\n\n"
            f"Current round: {attempt}; {int(attempt) - 1} earlier research rounds completed "
            "(including failures and unsubmitted rounds; baseline 000 excluded).\n"
            f"Current best: {best}; score: {score}. The draft is a copy of this version.\n"
            f"Available budget: up to {max_steps} model calls this round. "
            "You may submit whenever ready.",
        },
    ]


def parse_call(call):
    """Reject malformed native calls before they can enter the next API request."""
    if not isinstance(call, dict) or call.get("type") != "function":
        raise ValueError("Only function tools are supported")
    function = call.get("function")
    if (
        not isinstance(function, dict)
        or not isinstance(function.get("name"), str)
        or not function["name"]
    ):
        raise ValueError("Each tool call needs a function name")
    name = function["name"]
    if not isinstance(function.get("arguments"), str):
        raise ValueError(f"{name}: arguments must be a JSON object encoded as a string")
    try:
        args = json.loads(function["arguments"])
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{name}: invalid JSON at line {error.lineno}, column {error.colno} "
            f"(character {error.pos}): {error.msg}"
        ) from error
    if not isinstance(args, dict):
        raise ValueError(f"{name}: arguments must decode to an object, got {type(args).__name__}")
    return name, args


def attempt_info(directory, result):
    """Describe submitted versions and unevaluated drafts without inventing model intent."""
    submission = directory / "submission.json"
    kind = next((name for name in ("solver", "draft") if (directory / f"{name}.py").exists()), None)
    status = (
        "baseline"
        if directory.name == "000"
        else "submitted"
        if kind == "solver"
        else "unsubmitted"
    )
    summary, origin = None, None
    if submission.exists():
        summary, origin = json.loads(submission.read_text())["summary"], "model"
    elif status == "baseline":
        summary, origin = "Baseline solver.", "system"
    elif status == "unsubmitted":
        detail = result.get("error") or "No submit call completed"
        summary = f"Not submitted; draft not evaluated. {detail}"[:SUMMARY_LIMIT]
        origin = "system"
    return {
        "status": status,
        "source_kind": kind,
        "summary": summary,
        "summary_source": origin,
        **failure_info(
            result, default="not_submitted" if status == "unsubmitted" else "evaluation_error"
        ),
    }


def source_view(source, start_line=None, end_line=None):
    if start_line is None and end_line is None:
        return {"source": source}
    if source is None:
        raise ValueError("This attempt has no saved source to read")
    lines = list(io.StringIO(source, newline=""))
    if not lines:
        raise ValueError(
            "The source is empty (0 lines); omit the line range to read the empty source"
        )
    start = 1 if start_line is None else start_line
    end = len(lines) if end_line is None else end_line
    if start < 1 or start > len(lines):
        raise ValueError(f"start_line must be between 1 and {len(lines)}; received {start}")
    if end < start:
        raise ValueError(f"end_line must be >= start_line ({start}); received {end}")
    end = min(end, len(lines))
    return {
        "source": "".join(lines[start - 1 : end]),
        "start_line": start,
        "end_line": end,
        "total_lines": len(lines),
    }


def apply_solver_patch(source, patch):
    """Apply context hunks in memory, requiring unambiguous matches and no partial writes."""

    def split_lines(text):
        return text.replace("\r\n", "\n").removesuffix("\n").split("\n") if text else []

    lines, line_offset = split_lines(patch), 0
    if lines and lines[0] == "*** Begin Patch":
        if (
            len(lines) < 4
            or lines[1] != "*** Update File: solver.py"
            or lines[-1] != "*** End Patch"
        ):
            raise ValueError(
                "A wrapped patch must only update solver.py and end with *** End Patch"
            )
        lines = lines[2:-1]
        line_offset = 2
    if not lines:
        raise ValueError("patch must contain at least one @@ hunk")
    original, output, cursor, index, hunk = split_lines(source), [], 0, 0, 0

    def locate(context, start, end=False, label="context"):
        positions = (
            [len(original) - len(context)]
            if end
            else range(start, len(original) - len(context) + 1)
        )
        matches = [p for p in positions if p >= start and original[p : p + len(context)] == context]
        if len(matches) != 1:
            reason = "not found" if not matches else f"matches {len(matches)} locations"
            location = "at the end of solver.py" if end else f"from solver.py line {start + 1}"
            hint = "use read_solver and include exact, unique context"
            if end and not matches:
                elsewhere = [
                    p + 1
                    for p in range(start, len(original) - len(context) + 1)
                    if original[p : p + len(context)] == context
                ]
                hint = (
                    "*** End of File requires the hunk to end at EOF, not just terminate the patch"
                )
                if elsewhere:
                    hint += f"; this context starts at solver.py line(s) {elsewhere[:10]}"
                hint += "; use read_solver and omit the EOF marker when editing inside the file"
            raise ValueError(
                f"Patch hunk {hunk} (patch line {hunk_line}): {label} {reason} {location}; " + hint
            )
        return matches[0]

    while index < len(lines):
        hunk += 1
        hunk_line = index + line_offset + 1
        header = lines[index]
        if header != "@@" and not header.startswith("@@ "):
            raise ValueError(
                f"Patch hunk {hunk} (patch line {hunk_line}): each hunk must start with @@, "
                "optionally followed by an exact source line"
            )
        start = cursor
        if header != "@@" and not re.fullmatch(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@.*", header):
            start = locate([header[3:]], cursor, label="anchor") + 1
        index += 1
        old, new, changed, end = [], [], False, False
        while index < len(lines) and not lines[index].startswith("@@"):
            line = lines[index]
            index += 1
            if line == "*** End of File":
                end = True
                break
            if not line:
                line = " "
            if line[0] not in " +-":
                raise ValueError(
                    f"Patch hunk {hunk}, patch line {index + line_offset}: "
                    "lines must start with a space, "
                    "- or +; other files/operations are not allowed"
                )
            if line[0] in " -":
                old.append(line[1:])
            if line[0] in " +":
                new.append(line[1:])
            changed |= line[0] in "+-"
        if not changed:
            raise ValueError(
                f"Patch hunk {hunk} (patch line {hunk_line}): each hunk must add or remove lines"
            )
        if old:
            position = locate(old, start, end)
        elif original:
            raise ValueError(
                f"Patch hunk {hunk} (patch line {hunk_line}): "
                "insertions into a nonempty draft need unchanged context lines"
            )
        else:
            position = 0
        output.extend(original[cursor:position])
        output.extend(new)
        cursor = position + len(old)
    output.extend(original[cursor:])
    return "\n".join(output) + ("\n" if output else "")


class Workspace:
    """Only the draft is mutable; history is restricted to completed earlier attempts."""

    def __init__(self, root, source, ledger):
        self.root, self.source = root, source
        self.ledger = {}
        for row in ledger:
            self.ledger[row["attempt"]] = {**row, **attempt_info(root / row["attempt"], row)}
        self.submitted = False
        self.summary = None

    def history(self, order):
        if order not in ("recent", "score"):
            raise ValueError(f"order must be recent or score; received {order!r}")
        return sorted(
            self.ledger.values(),
            key=lambda row: (
                row["mean_score"]
                if order == "score" and row["valid"] and row["mean_score"] is not None
                else float("-inf"),
                int(row["attempt"]),
            ),
            reverse=True,
        )

    def page(self, rows, offset, limit):
        if offset < 0:
            raise ValueError(f"offset must be >= 0; received {offset}. Use 0 for the first page")
        if not 1 <= limit <= 50:
            raise ValueError(f"limit must be between 1 and 50; received {limit}")
        return {
            "total": len(rows),
            "attempts": [
                {k: row[k] for k in (*HISTORY_FIELDS, "matches") if k in row}
                for row in rows[offset : offset + limit]
            ],
        }

    def list_attempts(self, offset=0, limit=20, order="recent"):
        return self.page(self.history(order), offset, limit)

    def search_attempts(self, query, field="all", offset=0, limit=20, order="recent", mode="all"):
        if not query.strip():
            raise ValueError(
                "query must be nonblank; received an empty or whitespace-only string. "
                "Provide a keyword or phrase"
            )
        if "\n" in query or "\r" in query:
            raise ValueError("query must be a single-line keyword or phrase; remove line breaks")
        if field not in ("all", "summary", "code"):
            raise ValueError(f"field must be all, summary or code; received {field!r}")
        if mode not in ("all", "any", "literal"):
            raise ValueError(f"mode must be all, any or literal; received {mode!r}")
        terms = [query] if mode == "literal" else list(dict.fromkeys(query.split()))
        needles = [re.compile(re.escape(term), re.IGNORECASE) for term in terms]
        rows = []
        for row in self.history(order):
            fields = {"summary": row["summary"] or ""}
            if field != "summary" and row["source_kind"]:
                fields["code"] = (
                    self.root / row["attempt"] / f"{row['source_kind']}.py"
                ).read_text()
            matches, found = [], set()
            for name, text in fields.items():
                if field not in ("all", name):
                    continue
                for number, line in enumerate(io.StringIO(text, newline=""), 1):
                    line = line.rstrip("\r\n")
                    hits = [(i, needle.search(line)) for i, needle in enumerate(needles)]
                    hits = [(i, match) for i, match in hits if match]
                    found.update(i for i, _ in hits)
                    if hits and len(matches) < 3:
                        start = max(0, min(match.start() for _, match in hits) - 100)
                        snippet = line[start : start + 500]
                        matches.append({"field": name, "line": number, "text": snippet})
            if found and (mode != "all" or len(found) == len(needles)):
                rows.append({**row, "matches": matches})
        return {"query": query, "mode": mode, **self.page(rows, offset, limit)}

    def read_attempt(self, attempt, start_line=None, end_line=None):
        if attempt not in self.ledger:
            raise ValueError(
                f"Unknown previous attempt {attempt!r}; use an exact ID from list_attempts, "
                "including leading zeros (for example '003')"
            )
        directory = self.root / attempt
        kind = self.ledger[attempt]["source_kind"]
        return {
            **self.ledger[attempt],
            "parent": json.loads((directory / "decision.json").read_text())["parent"],
            **source_view(
                (directory / f"{kind}.py").read_text() if kind else None, start_line, end_line
            ),
        }

    def read_solver(self, start_line=None, end_line=None):
        return source_view(self.source, start_line, end_line)

    def load_solver(self, attempt):
        if attempt in ("latest", "latest_valid", "best"):
            matches = [
                row
                for row in self.history("recent")
                if attempt == "latest"
                or (attempt == "latest_valid" and row["valid"])
                or (attempt == "best" and row["decision"] in ("keep", "baseline"))
            ]
            if not matches:
                raise ValueError(
                    f"No previous attempt matches {attempt!r}; "
                    "use list_attempts to inspect available versions"
                )
            attempt = matches[0]["attempt"]
        record = self.read_attempt(attempt)
        if record["source"] is None:
            raise ValueError(
                f"Attempt {attempt!r} has no saved solver or draft to load; "
                "choose a version with source_kind solver or draft from list_attempts"
            )
        self.source = record["source"]
        return {"loaded": True, **record}

    def edit_solver(self, patch=None, source=None):
        if (patch is None) == (source is None):
            received = "neither field" if patch is None else "both fields"
            raise ValueError(
                "Provide exactly one of patch (context diff) or source (complete replacement); "
                f"received {received}"
            )
        updated = apply_solver_patch(self.source, patch) if patch is not None else source
        self.source = updated
        return {"edited": True}

    def submit(self, summary):
        if not summary.strip():
            raise ValueError(
                f"summary must be nonblank; received {len(summary)} characters, "
                f"empty or whitespace-only (maximum {SUMMARY_LIMIT}). "
                "Write a brief explanation of what changed and why"
            )
        if len(summary) > SUMMARY_LIMIT:
            raise ValueError(
                f"summary is too long: received {len(summary)} characters "
                f"(maximum {SUMMARY_LIMIT}; {len(summary) - SUMMARY_LIMIT} over). "
                "The count includes whitespace. Shorten summary and retry submit"
            )
        compile(self.source, "solver.py", "exec")
        self.submitted = True
        self.summary = summary.strip()
        return {"submitted": True, "summary": self.summary}

    def dispatch(self, call):
        name, schema = "tool call", None
        ignored_path = False
        try:
            if self.submitted:
                raise ValueError("Round already submitted; no further tools will run")
            name, args = parse_call(call)
            if name == "edit_solver" and "path" in args:
                args.pop("path")
                ignored_path = True
            schema = next((t["function"] for t in TOOLS if t["function"]["name"] == name), None)
            if schema is None:
                available = ", ".join(t["function"]["name"] for t in TOOLS)
                raise ValueError(f"Unknown tool; available tools: {available}")
            params = schema["parameters"]
            problems = []
            missing = sorted(set(params["required"]) - args.keys())
            unexpected = sorted(args.keys() - params["properties"].keys())
            if missing:
                problems.append(f"missing required arguments: {', '.join(missing)}")
            if unexpected:
                problems.append(f"unexpected arguments: {', '.join(unexpected)}")
            if missing or unexpected:
                allowed = ", ".join(params["properties"]) or "none (use {})"
                problems.append(f"allowed arguments: {allowed}")
            for key in sorted(args.keys() & params["properties"].keys()):
                value = args[key]
                kind = params["properties"][key]["type"]
                expected = str if kind == "string" else int
                if type(value) is not expected:
                    problems.append(f"{key} must be {kind}, got {type(value).__name__}")
                elif (
                    "enum" in params["properties"][key]
                    and value not in params["properties"][key]["enum"]
                ):
                    problems.append(
                        f"{key} must be one of {params['properties'][key]['enum']}; "
                        f"received {value!r}"
                    )
            if problems:
                raise ValueError("; ".join(problems))
            result = getattr(self, name)(**args)
        except (KeyError, TypeError, ValueError, SyntaxError) as error:
            detail = str(error)
            if isinstance(error, SyntaxError):
                location = "solver.py"
                if error.lineno is not None:
                    location += f" line {error.lineno}"
                if error.offset is not None:
                    location += f", column {error.offset}"
                detail = (
                    f"{error.msg} at {location}"
                    f"\n{(error.text or '').rstrip()}\n"
                    "Fix the draft with edit_solver, then retry submit; no code was executed"
                )
            if not self.submitted:
                detail += (
                    ". This call did not change the draft or submit it; the round is still open."
                )
            result = {"error": f"{name}: {type(error).__name__}: {detail}"}
            if schema is not None:
                result["expected_arguments"] = schema["parameters"]
        if ignored_path:
            result["notice"] = "Ignored path; edit_solver always edits the single draft solver.py."
        return result


def recover(messages, problem):
    messages.append(
        {
            "role": "user",
            "content": f"Your previous response could not be executed: {problem}. "
            "None of its tools ran; the draft is unchanged. Send complete function calls "
            "using the provided schemas. Large edits can be split across calls.",
        }
    )


def advance(response, workspace, messages):
    """Apply one complete assistant turn; also used to replay recorded tool actions."""
    finish = response["finish_reason"]
    if finish == "length":
        return recover(
            messages, "the response reached its output token limit (finish_reason=length)"
        )
    if finish not in ("stop", "tool_calls"):
        detail = f"; provider error: {json.dumps(response['error'])}" if "error" in response else ""
        return f"Model response did not finish: {finish}{detail}"
    calls = response.get("tool_calls") or []
    try:
        if not isinstance(calls, list):
            raise ValueError("tool_calls must be a list")
        for call in calls:
            parse_call(call)
        ids = [call.get("id") for call in calls]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("Tool calls require unique, nonempty IDs")
    except ValueError as error:
        return recover(messages, str(error))
    message = {"role": "assistant", "content": response["content"] or None}
    for key in ("tool_calls", "reasoning", "reasoning_details"):
        if response.get(key):
            message[key] = response[key]
    details = response.get("reasoning_details")
    if details:
        # Unsigned text deltas are one reasoning string, not separate reasoning blocks.
        plain = response.get("reasoning")
        if (
            plain
            and "".join(d.get("text", "") for d in details) == plain
            and all(
                d["type"] == "reasoning.text"
                and d.get("format") in (None, "unknown")
                and not d.get("id")
                and not d.get("signature")
                for d in details
            )
        ):
            message.pop("reasoning_details", None)
        else:
            message.pop("reasoning", None)
    messages.append(message)
    if not calls:
        return "Agent stopped without calling submit"
    mutation_failed = False
    for call in calls:
        name, _ = parse_call(call)
        if name == "submit" and mutation_failed:
            result = {
                "error": "submit: skipped because an earlier edit_solver or load_solver "
                "call in this response failed. Review the errors before submitting again; "
                "the round is still open."
            }
        else:
            result = workspace.dispatch(call)
        if name in ("edit_solver", "load_solver") and "error" in result and "attempt" not in result:
            mutation_failed = True
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
    return None


def propose(directory, manifest, best, ledger, key):
    max_steps = manifest["agent"]["max_steps"]
    workspace = Workspace(directory.parent, (best / "solver.py").read_text(), ledger)
    messages = initial_messages(
        (ROOT / f"tasks/{manifest['task']}/program.md").read_text(),
        best.name,
        json.loads((best / "evaluation.json").read_text())["mean_score"],
        max_steps,
        directory.name,
    )
    saved_steps = {p.name for p in (directory / "calls").glob("*") if p.is_dir()}
    if saved_steps != {f"{i:03d}" for i in range(1, len(saved_steps) + 1)}:
        raise RuntimeError("Saved model calls must be consecutive, starting with 001")
    if len(saved_steps) > max_steps:
        raise RuntimeError("Saved model calls exceed this round's recorded step limit")
    if not (directory / "agent.json").exists():
        (directory / "draft.py").write_text(workspace.source)
        save(directory / "agent.json", {"messages": messages, "submitted": False, "error": None})
    for step in range(1, max_steps + 1):
        call_dir = directory / "calls" / f"{step:03d}"
        call_dir.mkdir(parents=True, exist_ok=True)
        if step < len(saved_steps) and (
            not (call_dir / "response.json").exists()
            or inference.response_problem(json.loads((call_dir / "response.json").read_text()))
        ):
            raise RuntimeError(f"Only the last saved model call may be incomplete: {call_dir}")
        response = inference.complete(
            {
                "model": manifest["model"],
                "messages": messages,
                "tools": TOOLS,
                "stream": True,
                "stream_options": {"include_usage": True},
                **manifest["sampling"],
            },
            manifest["inference"]["endpoint"],
            key,
            call_dir,
        )
        previous = len(messages)
        error = advance(response, workspace, messages)
        if step < len(saved_steps) and (error or workspace.submitted):
            raise RuntimeError("Saved model calls continue after the round ended")
        if not workspace.submitted and not error and step == max_steps:
            error = f"Agent did not submit within {max_steps} model calls"
        if step >= len(saved_steps):
            (directory / "draft.py").write_text(workspace.source)
            save(
                directory / "agent.json",
                {"messages": messages, "submitted": workspace.submitted, "error": error},
            )
        print(
            f"  step {step}/{max_steps}: {error or ('submitted' if workspace.submitted else 'tools')}",
            flush=True,
        )
        for message in messages[previous:]:
            if message["role"] == "tool":
                result = json.loads(message["content"])
                if "error" in result and "attempt" not in result:
                    print(f"    {result['error']}", flush=True)
            elif message["role"] == "user":
                print(f"    {message['content']}", flush=True)
        if error or workspace.submitted:
            break
    if not workspace.submitted:
        return unsubmitted_result(error, max_steps)
    save(directory / "submission.json", {"summary": workspace.summary})
    (directory / "solver.py").write_text(workspace.source)
    return evaluate(workspace.source, manifest["seeds"], manifest["task"])


def unsubmitted_result(error, max_steps):
    kind = (
        "step_limit"
        if error == f"Agent did not submit within {max_steps} model calls"
        else "not_submitted"
        if error == "Agent stopped without calling submit"
        else "model_response_error"
    )
    return {"valid": False, "mean_score": None, "error": error, "failure_kind": kind}


def argument_parser():
    parser = research.argument_parser()
    parser.description = __doc__
    parser.set_defaults(max_tokens=65536)
    parser.add_argument(
        "--max-steps", type=int, default=128, help="Model calls per research round (default: 128)"
    )
    return parser


if __name__ == "__main__":
    research.run(argument_parser().parse_args(), agent=propose)
