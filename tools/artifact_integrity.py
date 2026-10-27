"""Original identities and transformed bytes for path-redacted run exports."""

import gzip
import hashlib
import json

from runner.evaluate import digest
from runner.search import structural_hash


class ArtifactIntegrity:
    def __init__(self, root):
        self.root = root
        self.exported = (root / "export.json").exists()
        self.entries = {}
        self.version = 0
        if self.exported:
            manifest = json.loads((root / "export.json").read_text())
            self.version = manifest["schema_version"] if "schema_version" in manifest else 1
            assert self.version in (1, 2)
            self.entries = {entry["file"]: entry for entry in manifest["files"]}
            assert len(self.entries) == len(manifest["files"])

    def verify_files(self):
        for name, entry in self.entries.items():
            path = self.root / name
            with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as stream:
                assert (
                    hashlib.file_digest(stream, "sha256").hexdigest() == entry["export_sha256"]
                ), name
            if path.name == "solver.py" and self.version == 2:
                self.fingerprint(name)

    def source_hash(self, name):
        if self.exported:
            return self.entries[name]["original_sha256"]
        return digest((self.root / name).read_text())

    def fingerprint(self, name):
        current = structural_hash((self.root / name).read_text())
        if self.version == 2:
            entry = self.entries[name]
            assert current == entry["export_structural_sha256"], name
            return entry["original_structural_sha256"]
        return current
