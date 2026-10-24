# Run an experiment

Baseline checks, reports and verification work without a model server. Searches use Flint locally by default, with optional remote HTTPS inference.

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

Flint is also available in [Ollama](https://ollama.com/library/flint). Its [v0.40.8-rc0 release](https://github.com/ollama/ollama/releases/tag/v0.40.8-rc0) supports Flint on macOS. The setup below uses `rec`.

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

## Another local model

For a local server with a streaming OpenAI-compatible chat API, pass its base URL and model ID:

```sh
uv run python research.py --task route --run runs/local-route --experiments 4 \
  --base-url http://127.0.0.1:8080/v1 --model your-model --max-tokens 16384
```

Use the same base URL and server-side model ID as your [OpenCode local provider](https://opencode.ai/docs/providers/#llamacpp). The runner calls that server directly. Replace `your-model` with the model you serve locally; `rec` is not needed for this path.

If the server needs authentication, add `--api-key-env LOCAL_MODEL_API_KEY` and set that environment variable. Only its name is saved. Keep it set when exporting the run so the credential check can use it.

## Remote model server

For a remote server with a streaming OpenAI-compatible chat API, configure its HTTPS base URL, model ID and API key environment variable. For example, with an OpenRouter API key set in `OPENROUTER_API_KEY`:

```sh
uv run python research.py --task circle --run runs/openrouter-circle --experiments 4 \
  --base-url https://openrouter.ai/api/v1 --model recursive/flint-v1.0:free \
  --api-key-env OPENROUTER_API_KEY
uv run python report.py runs/openrouter-circle
```

For another server, replace the URL, model ID and credential variable name. `--api-key-env` selects the variable explicitly; no provider or credential variable is inferred from the URL. Omit the flag for servers without authentication. Only the variable name is saved, not the key. Keep it set when exporting the run.

Remote URLs must use HTTPS; HTTP remains available for `localhost`, `127.0.0.1` and `::1`. URLs cannot contain credentials, query parameters or fragments. The runner appends `/chat/completions` to the base URL.

Task prompts, candidate code and evaluation feedback are sent to the selected server. Candidate execution and scoring still run in the local macOS sandbox. Set attempt and token limits appropriate to the server's capacity and pricing. Export and verification use the same commands as local runs.

## Start and inspect a search

Use one research run at a time, with no other client generating against the same local server.

```sh
uv run python research.py --task circle --run runs/circle-demo --experiments 4 --max-tokens 16384
uv run python report.py runs/circle-demo
```

Open `runs/circle-demo/report.html` for scores, solutions, responses and code changes. Every attempt counts, including failures and regressions. Each valid improvement becomes the starting point for the next attempt.

Each candidate gets 15 CPU seconds, including imports, and a 25-second wall limit. Generating the code takes additional time. Candidates run on CPU; model inference may use the GPU. The evaluator computes the score from the returned solution. Larger scores win; minimization tasks negate their cost. New runs use one fixed instance and evaluation seed 0.

The macOS sandbox blocks network access and child processes, restricts file access and uses a clean environment. Numerical-library thread limits are requested; a single native thread is not guaranteed.

Sampling uses temperature 1.0 and top-p 0.95, plus top-k 20 for the default Flint setup. Other servers receive only standard chat API fields. The command above allows up to 16,384 response tokens. The run manifest records the request settings and model connection.

## Resume

To extend a completed four-attempt run to eight total attempts:

```sh
uv run python research.py --run runs/circle-demo --resume --experiments 8
```

Resuming uses the saved seed and sampling settings and requires unchanged inputs, code, dependencies and local model configuration. Use a new run directory after changing them. Runs interrupted during an attempt cannot currently be resumed; keep their files and start a new run.

For another local or remote server, repeat the connection flags when resuming. Its URL, model ID and API key variable name are checked. Keep the loaded weights and server settings unchanged too; a hosted model ID alone does not freeze its underlying model version.

For bike demand, follow the [100-attempt protocol](../tasks/bike/README.md) and run `python -m tools.final_test` before exporting. It refits the frozen baseline and winner unchanged on training plus validation data and tests each once. It cannot be rerun, even if it fails.

## Save a run

Local `runs/` are ignored by Git. Export a finished run and verify it without executing candidate code:

```sh
uv run python -m tools.export_run runs/circle-demo examples/circle-demo
uv run python -m tools.verify_run examples/circle-demo
```

Exports include prompts, responses, timed streams, programs, scores and failures. They replace local paths with placeholders and check for the configured API credential (`rec auth token` for Flint, or the named environment variable for another server). Review the files before sharing. Annotations are separate notes; they are not sent to the model.

## Render a new bike run

After completing and verifying the run, rebuild the three-task overview with its bike measurements:

```sh
uv run python -m tools.render_gallery --bike-run runs/bike-demo --output-dir runs/bike-demo/media
```

This writes the overview GIF and still images to the selected directory, retaining the recorded circle and scheduling runs. It reads attempt counts, retained improvements and scores from the selected run. To reproduce the saved example with its inspected discovery labels:

```sh
uv run python -m tools.render_gallery \
  --bike-run examples/2026-10-21-bike \
  --bike-discoveries examples/2026-10-21-bike/discoveries.json \
  --output-dir runs/bike-gallery
```

For a new run, write a separate discoveries JSON from that run’s retained code changes and pass its filename with `--bike-discoveries`. The overview and detailed bike charts show up to three model hours by default; use `--bike-hours` to change that window for both. For the complete saved run, use `--bike-hours 6`. All attempts remain in the saved report. Rendering requires `ffmpeg` and does not call the model.
