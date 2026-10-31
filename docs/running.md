# Run an experiment

Baseline checks, reports and verification work without a model server. Searches use Flint locally by default.

These commands run one worker at a time. To run eight workers in parallel with llama.cpp, see [Eight workers](eight-workers.md).

## Setup

Use macOS and Python 3.12. Local MLX inference also needs an Apple Silicon Mac. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run these commands from this repository's root:

```sh
uv sync --frozen --python 3.12
uv run python -m tasks list
uv run python -m tasks check circle
```

The baseline check uses the research sandbox. See the [task catalog](../tasks/README.md) for other problems. To inspect a saved run, open its `report.html` in a browser or verify it:

```sh
uv run python -m tools.verify_run examples/2026-10-20-jobshop
```

Flint weights are available on Hugging Face: [full model](https://huggingface.co/RecursiveSI/flint) and [Mac INT4 weights](https://huggingface.co/RecursiveSI/flint-int4). The runner connects to the local `rec` engine. Follow the [official rec installation guide](https://docs.recursive.com/edge/rec). The [runtime source and releases](https://github.com/recursive-oss/rec) are public. You can inspect the saved experiments without installing the model.

Flint is also available in [Ollama](https://ollama.com/library/flint). The setup below uses `rec`.

### Install and start Flint

The rec engine requires an M4 Pro, M5 Pro or M5 Max, at least 48 GB of unified memory and about 30 GB of free disk space.

Install `rec` using the public release installer:

```sh
curl -fsSL https://raw.githubusercontent.com/recursive-oss/rec/main/install.sh | bash
```

Open a new terminal so `~/.local/bin` is on your PATH, then start the local server:

```sh
rec doctor
rec serve
```

The first `rec serve` downloads the model and runtime. Wait for `flint is ready` and leave that terminal open.

If you already have Flint installed, check that the server is ready before starting research:

```sh
rec status
```

The research loop connects directly to the local API, reading its endpoint and authentication token from `rec`.

Use `rec stop` when finished to free the model's memory. Runtime and model downloads are separate from this repository and have their own licensing terms.

## Another model

For another streaming OpenAI-compatible chat API, pass its base URL and model ID:

```sh
uv run python research.py --task route --run runs/local-route --experiments 4 \
  --base-url http://127.0.0.1:8080/v1 --model your-model --max-tokens 16384
```

For a local llama.cpp server, use its address and model alias, for example `--base-url http://127.0.0.1:8080/v1 --model flint`. The runner calls the server directly, so `rec` is not needed.

If the server needs authentication, add `--api-key-env LOCAL_MODEL_API_KEY` and set that environment variable. Only its name is saved. Keep it set when exporting the run.

For remote inference, use, for example, `--base-url https://openrouter.ai/api/v1 --model recursive/flint-v1.0:free --api-key-env OPENROUTER_API_KEY` with that variable set. Prompts, code and feedback are sent to the selected server; evaluation stays local.

Remote endpoints require HTTPS; HTTP is supported only on loopback hosts. The runner appends `/chat/completions` to the base URL. Embedded credentials, query parameters, fragments and redirects are rejected.

For serial and agent runs, the request layer retries interrupted streams, connection failures (including TLS interruptions), and HTTP or streamed provider errors with codes 408, 429, 500, 502, 503 and 504. It retries the identical request at most five times per invocation, waiting 30, 60, 120, 240 and 300 seconds. Failed requests and partial responses are preserved under that call's `retries/` directory and included in exports. No tools execute from these responses. Certificate verification, authentication and request errors are not retried; neither are completed responses, refusals or output-token limits. A permanent request error or exhausted retries stops at the current call, saves `failure.json`, and leaves the research round incomplete for `--resume`; it does not create an invalid candidate or consume a research round. Retries can incur additional provider usage, including usage not reported by an interrupted stream. Team workers use the same client with request retries disabled; their scheduler manages failures within the shared deadline.

## Start and inspect a search

Use one research run at a time, with no other client generating against the same local server.

```sh
uv run python research.py --task circle --run runs/circle-demo --experiments 4 --max-tokens 16384
uv run python report.py runs/circle-demo
```

Open `runs/circle-demo/report.html` for scores, solutions, responses and code changes. Every attempt counts, including failures and regressions. Each valid improvement becomes the starting point for the next attempt.

Each candidate gets 15 CPU seconds, including imports, and a 25-second wall limit. Generating the code takes additional time. Candidates run on CPU; model inference may use the GPU. The evaluator computes the score from the returned solution. Larger scores win; minimization tasks negate their cost. New runs use one fixed instance and evaluation seed 0.

The macOS sandbox blocks network access and child processes, restricts file access, uses a clean environment and sets numerical-library thread limits to one.

Sampling uses temperature 1.0 and top-p 0.95, plus top-k 20 for the default Flint setup. Other servers receive only standard chat API fields. The command above allows up to 16,384 response tokens. The run manifest records the request settings and model connection.

## Agent mode

`research_agent.py` reuses the model connection, evaluator, selection rule and run layout. It requires a chat server with native function tool calling. Use a new run directory:

```sh
uv run python research_agent.py --task circle --run runs/agent-circle --experiments 4 \
  --max-steps 128 --max-tokens 65536 \
  --base-url https://openrouter.ai/api/v1 --model recursive/flint-v1.0:free \
  --api-key-env OPENROUTER_API_KEY
uv run python report.py runs/agent-circle
```

Configure the server and credential variable as above. For a local server, use the same connection flags as `research.py`, or omit them for `rec`. The selected server must support tool calls; local runtime support depends on the server version.

Each round starts with a fresh conversation: the task from `program.md` without its one-shot instructions, the round number, completed-round count (including failures, excluding baseline 000), best attempt ID and score, and model-call budget. The editable draft starts from the best solver. Earlier conversations are omitted; history is available through tools. The model chooses its approach and action order, including full rewrites.

| Tool | Purpose |
| --- | --- |
| `list_attempts(order="recent", offset=0, limit=20)` | List earlier summaries, scores, decisions and failure reasons. `recent` is newest first; `score` is highest valid score first, breaking ties by recency and placing invalid/unscored attempts last. |
| `search_attempts(query, field="all", mode="all", order="recent", offset=0, limit=20)` | Search summaries and/or code (`all`, `summary`, `code`), including unsubmitted drafts. `all` requires every whitespace-separated term; `any` requires one; `literal` matches the entire phrase. |
| `read_attempt(attempt, start_line=..., end_line=...)` | Read a previous version's source, summary, parent and evaluation feedback, optionally by line range. |
| `load_solver(attempt)` | Replace the draft with an earlier solver or unevaluated draft. Accept an ID or `latest`, `latest_valid`, `best`; return the resolved ID, metadata and code. |
| `read_solver(start_line=..., end_line=...)` | Read the current draft, optionally by line range. |
| `edit_solver(patch=..., source=...)` | Provide exactly one: a context diff or the complete replacement source. Multiple hunks apply atomically. |
| `submit(summary)` | Compile-check Python, freeze the draft, evaluate it once and end the round. Require a nonblank explanation of at most 500 Unicode characters, including whitespace. |

History includes baseline 000 and completed failures, with up to 50 attempts per page. Records distinguish `baseline`, `submitted` and `unsubmitted` status, `solver` versus unevaluated `draft` source, and model summaries versus system notes. Measured `failure_kind` and `error_summary` are separate from the submission summary. Categories are `constraint_violation`, `invalid_output`, `runtime_error`, `timeout`, `step_limit`, `not_submitted`, `model_response_error` and the fallback `evaluation_error`; successful records have null failure fields. Invalid output is retained as `failed_output` in `evaluation.json`, outside compact model feedback.

Search is case-insensitive and uses literal substrings, not regexes. `all` terms can span lines or summary/code fields. Results echo `query` and `mode`, with up to three 500-character line snippets per attempt; no matches return `total: 0` and an empty list. Read ranges are 1-based and inclusive, defaulting to line 1 through EOF. They return actual bounds and `total_lines`, cap the end at EOF, and leave source text free of line-number prefixes.

For loading, `latest` selects the most recent completed round, including an unsubmitted draft; `latest_valid` selects the latest valid evaluation even if discarded; `best` selects the retained best. All include baseline 000. Loading replaces current edits without execution and fails if the selected source is missing. The recorded parent and improvement threshold still refer to the best at round start; the transcript records any loaded version.

The editor uses a single-file subset of [OpenAI's context-diff approach](https://developers.openai.com/api/docs/guides/tools-apply-patch). Pass a patch string such as:

```diff
@@
 def solve(n, seed, budget):
-    iterations = 100
+    iterations = 200
```

Prefix unchanged lines with a space, removals with `-`, additions with `+`; preserve source indentation. Separate hunks with `@@` and keep them in file order. `@@ exact source line` searches after that anchor; `*** End of File` requires the hunk to reach EOF. Numbered unified-diff headers are accepted, but matching uses exact, unique context. The optional `*** Begin Patch` / `*** Update File: solver.py` / `*** End Patch` wrapper is supported. Missing or ambiguous context rejects the whole edit. Patches normalize CRLF to LF and end nonempty files with a newline.

Only `solver.py` is editable. Omit `path`; if supplied, it is ignored with a notice. Other unsupported arguments produce errors. Successful edits return `{"edited": true}`. The tools expose no shell or arbitrary file access.

Within a round, messages and tool results accumulate normally. Tool-argument, summary and compilation errors leave the draft unchanged and the round open. Feedback gives the expected arguments, invalid bounds or options, summary character counts, or relevant patch/source location. If an edit or load fails, later submissions in that same response are skipped; earlier successful edits remain. The model can repair or submit the draft in a subsequent response.

Submission summaries explain what changed and why without a required template; surrounding whitespace is stripped when saved. Only submission executes code, under the same sandbox and 15 CPU / 25 wall seconds per seed. Results are available to the next round, and valid scores exceeding the best by more than 1e-9 are retained.

`--experiments` counts outer rounds, including failures. `--max-steps` (default 128) caps logical model calls per round; submission can happen earlier. Transport retries do not consume additional steps. `--max-tokens` caps each request, with an agent default of 65,536 (32,768 for `research.py`); choose a value supported by the model and remaining context.

Responses cut off by the output limit or containing malformed native calls execute no tools; a corrective message requests another response within the step budget. The raw response stays on disk. Stopping without `submit` or reaching the step limit records an invalid round and preserves the unevaluated draft. Inference failures instead stop at the current call for recovery, as described above.

Each round saves `draft.py`, `agent.json` (messages and tool results), and `calls/001/`, `calls/002/`, … with each request, response and stream. Submission adds `submission.json` and `solver.py`, followed by evaluation and decision records. Unsubmitted rounds get a system explanation capped at 500 characters. Reports show summaries, drafts, conversations and call links. Export verification reconstructs messages and replays tools to check source and summary provenance without executing candidates; use the original source version recorded in the manifest.

Resume with `research_agent.py --resume`, the original connection flags and desired `--experiments` total. Saved step/token limits are retained. Complete responses replay locally to rebuild the draft and conversation; generation continues at the first unfinished call. Saved requests must match reconstructed requests. The two entry points cannot resume each other's runs; the shared resume rules below also apply.

## Resume

To extend a completed four-attempt run to eight total attempts:

```sh
uv run python research.py --run runs/circle-demo --resume --experiments 8
```

Resuming uses the saved seed and sampling settings and requires unchanged inputs, code, dependencies and model configuration. Use a new run directory after changing them; this does not migrate runs made by older harness versions. The last incomplete attempt can resume in place. Complete saved responses are reused without new API requests, and a saved evaluation is finalized without rerunning the solver. If interruption occurred before the evaluation was saved, that same candidate must be evaluated again. JSON records are replaced atomically, and completed decisions rebuild the summary when resuming.

For another local or remote server, repeat the connection flags when resuming. Its URL, model ID and API key variable name are checked. Keep the loaded weights and server settings unchanged too.

For bike demand, follow the [100-attempt protocol](../tasks/bike/README.md) and run `python -m tools.final_test` before exporting. It refits the frozen baseline and winner unchanged on training plus validation data and tests each once. It cannot be rerun, even if it fails.

## Save a run

Local `runs/` are ignored by Git. Export a finished run and verify it without executing candidate code:

```sh
uv run python -m tools.export_run runs/circle-demo examples/circle-demo
uv run python -m tools.verify_run examples/circle-demo
```

Exports include prompts, responses, timed streams, programs, scores and failures. They replace local paths with placeholders and stop if the API key named by `--api-key-env` appears in any file. Review the files before sharing.

## Render a new bike run

To rebuild the three-task overview with your own bike run:

```sh
uv run python -m tools.render_gallery --bike-run runs/bike-demo --output-dir runs/bike-demo/media
```

Add `--in-progress` to preview a run that is still going, and `--still-only` to skip the GIF. To label improvements on the bike chart, pass a JSON file with `--bike-discoveries`, like the [saved example](../examples/2026-10-21-bike/discoveries.json). Each entry has `attempt`, `source_sha256`, `label` and `position`, given as fractions of the plot area. The charts show the first three model hours; use `--bike-hours` to change that. Rendering needs `ffmpeg` and doesn't call the model.
