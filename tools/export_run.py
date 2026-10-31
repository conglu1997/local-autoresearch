"""Export a run with path redaction, credential checks and compressed streams."""

from contextlib import ExitStack
import argparse
import gzip
import hashlib
import json
import os
import re
from pathlib import Path

from report import report
from runner.inference import CALL_FILES
from runner.search import structural_hash

ROOT_FILES = {
    "runtime.json",
    "runtime-memory.jsonl",
    "journal.jsonl",
    "manifest.json",
    "finished.json",
    "summary.json",
    "best.py",
    "validation.json",
    "notes.json",
    "annotations.json",
    "continuation.json",
    "progress.json",
    "final-test-started.json",
    "final-test.json",
    "baseline-final-test.json",
}
ATTEMPT_FILES = CALL_FILES | {
    "agent.json",
    "draft.py",
    "submission.json",
    "retry.json",
    "admission.json",
    "completed.json",
    "worker.log",
    "interrupted.json",
    "solver.py",
    "evaluation.json",
    "decision.json",
    "diagnostic.json",
}


def redactions(text):
    pattern = (
        rf"(?P<home>{re.escape(str(Path.home()))})"
        r"|(?P<workspace>/private/var/folders/[^\s\"'\\]+/flint-packing-[^/\s\"'\\]+)"
        r"|(?P<volume>(?:/Volumes/|(?<![\w/])Volumes/)[^\s\"'\\]+)"
    )
    for match in re.finditer(pattern, text):
        if match.lastgroup == "volume":
            replacement = (
                "<ABSOLUTE_PATH>/" if match[0].startswith("/") else "<RELATIVE_PATH>/"
            ) + match[0].rsplit("/", 1)[-1]
        else:
            replacement = "<HOME>" if match.lastgroup == "home" else "<WORKSPACE>"
        yield match.start(), match.end(), replacement


def redact(text):
    parts, offset = [], 0
    for start, stop, replacement in redactions(text):
        parts.extend((text[offset:start], replacement))
        offset = stop
    return "".join(parts) + text[offset:]


def reject_credentials(text, key, filename):
    if key is not None and key in text:
        raise RuntimeError(f"Credential detected in {filename}; export stopped")


def redacted_stream(lines, key=None):
    """Redact assembled model fields, then distribute edits across their original chunks."""
    rows = []
    fields = {}
    for line in lines:
        event = json.loads(line)
        chunk = json.loads(event["line"][6:]) if event["line"].startswith("data: {") else None
        rows.append((line, event, chunk, json.dumps(chunk)))
        if chunk is None or "error" in chunk:
            continue
        for choice in chunk.get("choices", []):
            delta = choice["delta"]
            for name in ("content", "reasoning", "reasoning_content"):
                if delta.get(name):
                    fields.setdefault((choice.get("index", 0), name), []).append((delta, name))
            for tool in delta.get("tool_calls") or []:
                function = tool.get("function") or {}
                for name in ("name", "arguments"):
                    if function.get(name):
                        fields.setdefault((choice.get("index", 0), tool["index"], name), []).append(
                            (function, name)
                        )
    for fragments in fields.values():
        original = "".join(obj[name] for obj, name in fragments)
        reject_credentials(original, key, "stream.jsonl")
        edits = list(redactions(original))
        if not edits:
            continue
        offset = 0
        for obj, name in fragments:
            end = offset + len(obj[name])
            parts, cursor = [], offset
            for start, stop, replacement in edits:
                if stop <= offset or start >= end:
                    continue
                parts.append(original[cursor : max(cursor, start)])
                if offset <= start:
                    parts.append(replacement)
                cursor = min(stop, end)
            obj[name] = "".join(parts) + original[cursor:end]
            offset = end
    for original, event, chunk, before in rows:
        if json.dumps(chunk) == before:
            yield redact(original)
        else:
            event["line"] = "data: " + json.dumps(chunk, ensure_ascii=False)
            yield redact(json.dumps(event, ensure_ascii=False)) + "\n"


def export(source, target):
    manifest = json.loads((source / "manifest.json").read_text())
    name = manifest["inference"].get("api_key_env")
    key = os.environ[name] if name else None
    target.mkdir(parents=True, exist_ok=False)
    entries = []
    paths = [source / name for name in ROOT_FILES if (source / name).exists()]
    paths += [
        p
        for d in source.iterdir()
        if d.is_dir() and d.name.isdigit()
        for p in d.rglob("*")
        if p.is_file()
        if p.name in ATTEMPT_FILES
    ]
    for path in sorted(paths):
        relative = path.relative_to(source)
        destination = target / relative
        compressed = path.name == "stream.jsonl"
        if compressed:
            destination = destination.with_suffix(".jsonl.gz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        original_hash, export_hash = hashlib.sha256(), hashlib.sha256()
        redacted = False
        with ExitStack() as stack:
            incoming = stack.enter_context(path.open())
            outgoing = stack.enter_context(destination.open("wb"))
            if compressed:
                outgoing = stack.enter_context(
                    gzip.GzipFile(filename="", mode="wb", fileobj=outgoing, mtime=0)
                )
            if compressed:
                lines = list(incoming)
                paired = zip(lines, redacted_stream(lines, key))
            else:
                paired = ((line, redact(line)) for line in incoming)
            for original, text in paired:
                reject_credentials(original, key, path.name)
                original_hash.update(original.encode())
                export_hash.update(text.encode())
                redacted |= text != original
                outgoing.write(text.encode())
        entry = {
            "file": str(destination.relative_to(target)),
            "original_sha256": original_hash.hexdigest(),
            "export_sha256": export_hash.hexdigest(),
            "path_redacted": redacted,
        }
        if path.name == "solver.py":
            entry["original_structural_sha256"] = structural_hash(path.read_text())
            entry["export_structural_sha256"] = structural_hash(destination.read_text())
        entries.append(entry)
    (target / "export.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "transformations": [
                    "Home directory → <HOME>",
                    "Candidate temporary directory → <WORKSPACE>",
                    "Mounted-volume paths → <ABSOLUTE_PATH> or <RELATIVE_PATH>, retaining the filename",
                    "Raw streams use lossless gzip after path redaction",
                ],
                "files": entries,
            },
            indent=2,
        )
        + "\n"
    )
    report(target)
    print(f"Exported {len(entries)} files to {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    export(args.source.resolve(), args.target.resolve())
