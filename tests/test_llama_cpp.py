"""Runtime setup and supervision with downloads, builds and processes faked."""

from argparse import Namespace
from contextlib import contextmanager
import hashlib
import io
import json

import pytest

from tools import llama_cpp

GIB = 1024**3


class Download(io.BytesIO):
    def __init__(self, data, status):
        super().__init__(data)
        self.status = status


def test_download_resumes_a_partial_file(tmp_path, monkeypatch):
    data = b"pinned model bytes"
    target = tmp_path / "model.gguf"
    (tmp_path / "model.gguf.download").write_bytes(data[:6])
    ranges = []

    def urlopen(request, timeout):
        ranges.append(request.get_header("Range"))
        return Download(data[6:], 206)

    monkeypatch.setattr(llama_cpp, "urlopen", urlopen)
    llama_cpp.download("https://example.test/model", target, hashlib.sha256(data).hexdigest())
    assert ranges == ["bytes=6-"]
    assert target.read_bytes() == data


def test_download_rejects_a_wrong_file(tmp_path, monkeypatch):
    monkeypatch.setattr(llama_cpp, "urlopen", lambda request, timeout: Download(b"bad", 200))
    with pytest.raises(ValueError, match="SHA-256"):
        llama_cpp.download("https://example.test/model", tmp_path / "model.gguf", "0" * 64)


def test_setup_reuses_a_finished_build(tmp_path, monkeypatch):
    binary = tmp_path / f"source/llama.cpp-{llama_cpp.COMMIT}/build/bin/llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_text("built")
    downloads = []
    monkeypatch.setattr(llama_cpp, "download", lambda url, target, sha: downloads.append(target))
    monkeypatch.setattr(llama_cpp.subprocess, "run", lambda *a, **k: pytest.fail("rebuilt"))
    llama_cpp.setup(tmp_path)
    assert downloads == [tmp_path / llama_cpp.MODEL_FILE]
    receipt = llama_cpp.installation(tmp_path)
    command = llama_cpp.command(tmp_path, receipt, 11501, 8)
    assert command[0] == str(binary)
    assert command[command.index("-np") + 1] == "8"
    assert command[command.index("-c") + 1] == str(8 * 65536)


def test_run_stops_server_and_research_when_memory_runs_out(tmp_path, monkeypatch):
    signals = []

    class Process:
        pid = 4242
        returncode = None

        def __init__(self, argv, **kwargs):
            pass

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = -15

    @contextmanager
    def props(url, timeout):
        yield io.BytesIO(json.dumps({"total_slots": 8}).encode())

    samples = iter([0, 1, 1, 50])
    monkeypatch.setattr(llama_cpp, "installation", lambda directory: {"binary": "llama-server"})
    monkeypatch.setattr(llama_cpp, "ready", lambda endpoint: True)
    monkeypatch.setattr(llama_cpp, "urlopen", props)
    monkeypatch.setattr(
        llama_cpp, "memory", lambda: {"rss": next(samples) * GIB, "footprint": 0, "swap": 0}
    )
    monkeypatch.setattr(llama_cpp.subprocess, "Popen", Process)
    monkeypatch.setattr(llama_cpp.os, "killpg", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(llama_cpp.signal, "signal", lambda *args: None)
    monkeypatch.setattr(llama_cpp.time, "sleep", lambda seconds: None)
    args = Namespace(
        directory=tmp_path,
        run=tmp_path / "run",
        task="circle",
        hours=1,
        workers=8,
        replicate=0,
        port=0,
    )
    with pytest.raises(SystemExit, match="more than 42 GiB"):
        llama_cpp.run(args)
    assert len(signals) == 2
    runtime = json.loads((tmp_path / "run.runtime/runtime.json").read_text())
    assert runtime["server_props"] == {"total_slots": 8}
