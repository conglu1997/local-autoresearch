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

## Resume

To extend a completed four-attempt run to eight total attempts:

```sh
uv run python research.py --run runs/circle-demo --resume --experiments 8
```

Resuming uses the saved seed and sampling settings and requires unchanged inputs, code, dependencies and model configuration. Use a new run directory after changing them. If a run stopped partway through an attempt, resuming moves that attempt aside to `NNN-interrupted-1` and runs it again.

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
