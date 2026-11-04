"""Materialized expression data with aligned physical disk reads, no page-cache fallback."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from gpu_pilot.config import load_config
from gpu_pilot.synthetic import SyntheticExpressionDataset, SyntheticExpressionSpec
from gpu_pilot.zarr_stream import thread_read_bytes


def record_size(features: int) -> int:
    return ((features * 9 + 4095) // 4096) * 4096


class ScratchExpressionDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, root: Path, spec: SyntheticExpressionSpec):
        self.root, self.spec = root, spec
        manifest = json.loads((root / "manifest.json").read_text())
        for key in ("seed", "feature_count", "latent_size", "masking_rate", "sparsity_rate"):
            if manifest["spec"][key] != getattr(spec, key):
                raise ValueError(f"Scratch expression metadata mismatch: {key}")
        if spec.sample_offset + spec.dataset_size > manifest["spec"]["dataset_size"]:
            raise ValueError("Scratch expression sample range exceeds materialized data")
        if not (root / "_SUCCESS").exists():
            raise ValueError("Incomplete expression dataset")

    def __len__(self) -> int:
        return self.spec.dataset_size

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        size, features = record_size(self.spec.feature_count), self.spec.feature_count
        before = thread_read_bytes()
        fd = os.open(self.root / "expression.bin", os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW)
        try:
            with mmap.mmap(-1, size) as buffer:
                count = os.preadv(fd, [buffer], (index + self.spec.sample_offset) * size)
                if count != size or thread_read_bytes() - before < size:
                    raise OSError("Expression physical-read verification failed")
                x = np.frombuffer(buffer, dtype=np.float32, count=features).copy()
                y = np.frombuffer(buffer, dtype=np.float32, count=features, offset=features * 4).copy()
                mask = np.frombuffer(buffer, dtype=np.bool_, count=features, offset=features * 8).copy()
        finally:
            os.close(fd)
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(mask)


def write_chunk(task: tuple[str, SyntheticExpressionSpec, int, int]) -> int:
    path, spec, start, stop = task
    torch.set_num_threads(1)
    dataset = SyntheticExpressionDataset(spec)
    size = record_size(spec.feature_count)
    fd = os.open(path, os.O_WRONLY)
    try:
        for index in range(start, stop):
            values = dataset[index]
            raw = b"".join(value.numpy().tobytes() for value in values)
            raw += bytes(size - len(raw))
            if os.pwrite(fd, raw, index * size) != size:
                raise OSError("Short expression write")
    finally:
        os.close(fd)
    return stop - start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config.synthetic
    assert settings is not None
    spec = SyntheticExpressionSpec(config.run.seed, settings.dataset_size, settings.feature_count,
                                   settings.latent_size, settings.masking_rate, settings.sparsity_rate)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / "expression.bin"
    with path.open("wb") as handle:
        handle.truncate(record_size(spec.feature_count) * len(SyntheticExpressionDataset(spec)))
    tasks = [(str(path), spec, start, min(start + 1000, spec.dataset_size))
             for start in range(0, spec.dataset_size, 1000)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for count, _ in enumerate(pool.map(write_chunk, tasks), 1):
            if count % 10 == 0:
                print(json.dumps({"event": "expression_materialize", "chunks": count}), flush=True)
    with path.open("rb") as handle:
        os.fsync(handle.fileno())
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = {"spec": asdict(spec), "sha256": digest, "bytes": path.stat().st_size,
                "record_bytes": record_size(spec.feature_count), "storage": "O_DIRECT"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (args.output / "checksums.sha256").write_text(f"{digest}  expression.bin\n")
    (args.output / "_SUCCESS").write_text(json.dumps(manifest))
    dataset = ScratchExpressionDataset(args.output, spec)
    original = SyntheticExpressionDataset(spec)
    for index in (0, 1, 89999, 90000, 99999):
        for actual, expected in zip(dataset[index], original[index], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    print(json.dumps({"event": "expression_verified", **manifest}), flush=True)


if __name__ == "__main__":
    main()
