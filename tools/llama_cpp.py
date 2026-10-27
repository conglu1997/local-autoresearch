"""Install the pinned llama.cpp and Flint model, then run a team of workers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

from runner.core import save
from runner.evaluate import ROOT
from runner.resources import limit_reason, memory
from runner.teams import WORKERS
from tasks import TASKS

COMMIT = "0ca39723762a5ce71ef57cb6f3b509f377a867fe"
ARCHIVE_SHA = "14361cfe1af9dd07c7bd9db79cf5196e6fe67cebcd0e46cd8b60cfecf9962388"
MODEL_REVISION = "9c47151036bc2c34331142dda04488bceab2bc9f"
MODEL_SHA = "43c07baccdba42eb7528a9630ee15699b1942c297291dc1b39a061f508ee1a4b"
MODEL_FILE = "flint-int4-q4_1.gguf"
MODEL_URL = (
    f"https://huggingface.co/RecursiveSI/flint-int4-gguf/resolve/{MODEL_REVISION}/{MODEL_FILE}"
)
PATCH = Path(__file__).resolve().parent / "patches" / "llama-jinja-hex.patch"
CONTEXT = 65536


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, target, expected):
    """Download with progress, resuming a partial file, then check its SHA-256."""
    if not target.exists():
        partial = target.with_name(target.name + ".download")
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"Range": f"bytes={start}-"} if start else {}
        with urlopen(Request(url, headers=headers), timeout=60) as incoming:
            done = start if incoming.status == 206 else 0
            with partial.open("r+b" if done else "wb") as outgoing:
                outgoing.seek(done)
                while chunk := incoming.read(1 << 27):
                    outgoing.write(chunk)
                    done += len(chunk)
                    print(f"\r{target.name}: {done / 1e9:.1f} GB", end="", flush=True)
        print()
        partial.replace(target)
    if sha(target) != expected:
        raise ValueError(f"{target} does not match its pinned SHA-256; delete it and run again")


def build(directory):
    code = directory / "source" / f"llama.cpp-{COMMIT}"
    binary = code / "build/bin/llama-server"
    if binary.exists():
        return binary
    archive = directory / "llama.cpp.tar.gz"
    download(
        f"https://codeload.github.com/ggml-org/llama.cpp/tar.gz/{COMMIT}", archive, ARCHIVE_SHA
    )
    if code.exists():
        shutil.rmtree(code)
    with tarfile.open(archive) as bundle:
        bundle.extractall(directory / "source", filter="data")
    subprocess.run(["patch", "-p1", "-i", str(PATCH)], cwd=code, check=True)
    subprocess.run(
        [
            "cmake",
            "-S",
            str(code),
            "-B",
            str(code / "build"),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DGGML_METAL=ON",
            "-DLLAMA_CURL=OFF",
            "-DLLAMA_BUILD_TESTS=OFF",
        ],
        check=True,
    )
    subprocess.run(
        ["cmake", "--build", str(code / "build"), "--target", "llama-server", "-j", "8"],
        check=True,
    )
    return binary


def setup(directory):
    directory.mkdir(parents=True, exist_ok=True)
    binary = build(directory)
    download(MODEL_URL, directory / MODEL_FILE, MODEL_SHA)
    save(
        directory / "installation.json",
        {
            "commit": COMMIT,
            "patch_sha256": sha(PATCH),
            "model_revision": MODEL_REVISION,
            "model_sha256": MODEL_SHA,
            "binary": str(binary.relative_to(directory)),
        },
    )
    print(f"Installed llama.cpp {COMMIT[:7]} and Flint in {directory}")


def installation(directory):
    receipt = json.loads((directory / "installation.json").read_text())
    if (receipt["commit"], receipt["patch_sha256"], receipt["model_revision"]) != (
        COMMIT,
        sha(PATCH),
        MODEL_REVISION,
    ):
        raise ValueError("This runtime is from another version; run setup in a new directory")
    return receipt


def command(directory, receipt, port, workers):
    return [
        str(directory / receipt["binary"]),
        "-m",
        str(directory / MODEL_FILE),
        "--alias",
        "flint",
        "-ngl",
        "all",
        "-c",
        str(workers * CONTEXT),
        "-np",
        str(workers),
        "--cont-batching",
        "--cache-ram",
        "0",
        "-b",
        "256",
        "-ub",
        "128",
        "-t",
        "8",
        "-fa",
        "on",
        "-ctk",
        "q8_0",
        "-ctv",
        "q8_0",
        "--jinja",
        "--temp",
        "1",
        "--top-p",
        "0.95",
        "--top-k",
        "20",
        "--min-p",
        "0",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


def check_port(port):
    """Fail early if another server is listening on the port."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))


def ready(endpoint):
    try:
        with urlopen(endpoint + "/health", timeout=5):
            return True
    except (URLError, ConnectionError):
        return False


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run(args):
    logs = args.run.with_name(args.run.name + ".runtime")
    if args.run.exists() or logs.exists():
        sys.exit(f"{args.run} or {logs} already exists; choose a new run directory")
    receipt = installation(args.directory)
    check_port(args.port)
    endpoint = f"http://127.0.0.1:{args.port}"
    runtime = {
        "installation": receipt,
        "server_command": command(args.directory, receipt, args.port, args.workers),
        "research_command": [
            sys.executable,
            str(ROOT / "research.py"),
            "--task",
            args.task,
            "--run",
            str(args.run.resolve()),
            "--hours",
            str(args.hours),
            "--workers",
            str(args.workers),
            "--replicate",
            str(args.replicate),
            "--base-url",
            endpoint + "/v1",
            "--model",
            "flint",
        ],
    }
    logs.mkdir(parents=True)
    save(logs / "runtime.json", runtime)

    stopping = False

    def interrupt(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    initial_swap = memory()["swap"]
    research = None
    with (
        (logs / "server.log").open("w") as log,
        (logs / "server.log").open() as server_output,
        (logs / "memory.jsonl").open("w") as samples,
    ):
        server = subprocess.Popen(
            runtime["server_command"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        recent_output = ""

        def problem():
            nonlocal recent_output
            sample = memory()
            samples.write(json.dumps(sample) + "\n")
            samples.flush()
            recent_output = recent_output[-200:] + server_output.read()
            if stopping:
                return "interrupted"
            if server.poll() is not None:
                return "llama.cpp exited; see server.log"
            if "kIOGPUCommandBufferCallbackErrorOutOfMemory" in recent_output:
                return "llama.cpp ran out of GPU memory; see server.log"
            return limit_reason(sample, initial_swap)

        try:
            while not (reason := problem()) and not ready(endpoint):
                time.sleep(1)
            if not reason:
                with urlopen(endpoint + "/props", timeout=5) as response:
                    save(logs / "runtime.json", {**runtime, "server_props": json.load(response)})
                research = subprocess.Popen(runtime["research_command"], start_new_session=True)
                while research.poll() is None and not (reason := problem()):
                    time.sleep(2)
        finally:
            if research is not None:
                stop(research)
            stop(server)
    if args.run.exists():
        shutil.copy2(logs / "runtime.json", args.run / "runtime.json")
        shutil.copy2(logs / "memory.jsonl", args.run / "runtime-memory.jsonl")
    if reason:
        sys.exit(f"Stopped: {reason}. Logs are in {logs}")
    if research.returncode:
        sys.exit(f"research.py exited with code {research.returncode}. Logs are in {logs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    install = actions.add_parser("setup", help="Download and build the pinned runtime")
    launch = actions.add_parser("run", help="Start llama.cpp and run a team of workers")
    for action in (install, launch):
        action.add_argument("--directory", type=Path, required=True, help="Runtime directory")
    launch.add_argument("--run", type=Path, required=True)
    launch.add_argument("--task", choices=list(TASKS), default="circle")
    launch.add_argument("--hours", type=float, required=True)
    launch.add_argument("--workers", type=int, default=WORKERS)
    launch.add_argument("--replicate", type=int, default=0)
    launch.add_argument("--port", type=int, default=11501)
    args = parser.parse_args()
    args.directory = args.directory.resolve()
    if args.action == "setup":
        setup(args.directory)
    else:
        run(args)


if __name__ == "__main__":
    main()
