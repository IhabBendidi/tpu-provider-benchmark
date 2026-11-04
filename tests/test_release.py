"""A reorganized checkout must still stage the established worker filesystem."""

import hashlib
import json
import tarfile
from pathlib import Path

from submit import write_release_archive

ROOT = Path(__file__).resolve().parents[1]


def test_release_preserves_worker_paths_and_frozen_dependencies(tmp_path):
    archive = tmp_path / "release.tar.gz"
    write_release_archive(archive)
    with tarfile.open(archive) as release:
        names = set(release.getnames())
        expected = {
            "historical_benchmark.py": ROOT / "src/tpu_benchmark/historical_benchmark.py",
            "smoke-worker.sh": ROOT / "scripts/slurm/smoke-worker.sh",
            "phase-worker.sh": ROOT / "scripts/slurm/phase-worker.sh",
            "single-trial-worker.sh": ROOT / "scripts/slurm/single-trial-worker.sh",
            "bootstrap-worker.sh": ROOT / "scripts/setup/bootstrap-worker.sh",
            "config/vit-4chip.toml": ROOT / "configs/vit-4chip.toml",
            "config/rpe1-selection.json": ROOT / "configs/rpe1-selection.json",
            "requirements/xla.txt": ROOT / "requirements/xla.txt",
            "requirements/torchtpu.txt": ROOT / "requirements/torchtpu.txt",
        }
        for name, source in expected.items():
            assert release.extractfile(name).read() == source.read_bytes()
        manifest = json.loads(release.extractfile("baseline/provenance.json").read())
        for name, digest in manifest["files"].items():
            data = release.extractfile("baseline/" + name).read()
            assert hashlib.sha256(data).hexdigest() == digest
        assert not any("__pycache__" in name or name.startswith("test_") for name in names)
