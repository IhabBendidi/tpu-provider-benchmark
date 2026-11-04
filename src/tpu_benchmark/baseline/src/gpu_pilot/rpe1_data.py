"""Pinned, leakage-audited RPE1 preprocessing and portable compressed data staging."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
from scipy import sparse  # type: ignore[import-untyped]

SOURCE_URL = "https://zenodo.org/records/13350497/files/ReplogleWeissman2022_rpe1.h5ad?download=1"
SOURCE_MD5 = "cc7f1ec50aeb3a3e1b4a6cfa713d80fa"
PARTITIONS = ("train", "validation", "test")


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strings(node: h5py.Group | h5py.Dataset) -> np.ndarray:
    if isinstance(node, h5py.Group):
        categories = strings(node["categories"])
        codes = node["codes"][:]
        if np.any(codes < 0):
            raise ValueError("Missing categorical metadata requires an explicit exclusion rule")
        return np.asarray(categories[codes])
    return np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in node[:]])


def make_splits(labels: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Labels -1 are controls; other labels are sorted target-gene IDs.

    Perturbation cohorts are disjoint. Cell partitions are also disjoint, while
    10% support cells from each validation/test cohort intentionally enter train.
    """
    roles = np.full(len(labels), -1, dtype=np.int8)
    partitions = np.full(len(labels), -1, dtype=np.int8)
    support = np.zeros(len(labels), dtype=bool)
    groups, counts = np.unique(labels[labels >= 0], return_counts=True)
    if np.any(counts < 2):
        raise ValueError("Perturbations with fewer than two cells must be excluded explicitly")
    order = np.random.default_rng(np.random.SeedSequence([seed, 0])).permutation(groups)
    n_test, n_val = int(len(groups) * 0.20), int(len(groups) * 0.10)
    if min(n_test, n_val) < 1:
        raise ValueError("Too few perturbations for the requested cohorts")
    cohorts = {
        int(g): 2 if i < n_test else 1 if i < n_test + n_val else 0 for i, g in enumerate(order)
    }
    for group in groups:
        ids = np.flatnonzero(labels == group)
        ids = np.random.default_rng(np.random.SeedSequence([seed, 1, int(group)])).permutation(ids)
        role = cohorts[int(group)]
        roles[ids] = role
        partitions[ids] = role
        if role:
            n_support = min(len(ids) - 1, max(1, math.ceil(len(ids) * 0.10)))
            partitions[ids[:n_support]] = 0
            support[ids[:n_support]] = True
    controls = np.flatnonzero(labels == -1)
    controls = np.random.default_rng(np.random.SeedSequence([seed, 2])).permutation(controls)
    n_test, n_val = int(len(controls) * 0.20), int(len(controls) * 0.10)
    if min(n_test, n_val) < 1:
        raise ValueError("Too few controls for disjoint control pools")
    partitions[controls[:n_test]] = 2
    partitions[controls[n_test : n_test + n_val]] = 1
    partitions[controls[n_test + n_val :]] = 0
    assert np.all(partitions >= 0)
    return partitions, roles, support


def normalize_selected(counts: np.ndarray) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float32)
    totals = values.sum(axis=1, dtype=np.float64)
    if np.any(totals <= 0) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Selected-gene libraries must be positive, finite and nonnegative")
    return np.log1p(values * (3000.0 / totals).astype(np.float32)[:, None]).astype(np.float32)


def csr_rows(node: h5py.Group | h5py.Dataset, start: int, stop: int) -> sparse.csr_matrix:
    if isinstance(node, h5py.Dataset):
        values = node[start:stop]
        if (
            np.any(values < 0)
            or not np.all(np.isfinite(values))
            or not np.all(values == np.floor(values))
        ):
            raise ValueError("Source X is not nonnegative integer raw counts")
        return sparse.csr_matrix(values)
    pointers = node["indptr"][start : stop + 1]
    a, b = int(pointers[0]), int(pointers[-1])
    values = node["data"][a:b]
    if (
        np.any(values < 0)
        or not np.all(np.isfinite(values))
        or not np.all(values == np.floor(values))
    ):
        raise ValueError("Source X is not nonnegative integer raw counts")
    return sparse.csr_matrix(
        (values, node["indices"][a:b], pointers - pointers[0]),
        shape=(stop - start, int(node.attrs["shape"][1])),
    )


def prepare(source: Path, package: Path, local: Path, seed: int, genes_count: int = 2000) -> None:
    started = time.perf_counter()
    if (package / "manifest.json").exists():
        raise ValueError("Prepared dataset already exists; choose a new output to change the split")
    if digest(source, "md5") != SOURCE_MD5:
        raise ValueError("Source checksum does not match the pinned public dataset")
    package.mkdir(parents=True, exist_ok=True)
    local.mkdir(parents=True, exist_ok=True)
    with h5py.File(source) as f:
        if f["X"].attrs.get("encoding-type") not in ("csr_matrix", "array"):
            raise ValueError("Expected dense or sparse CSR raw-count X")
        obs, var, x = f["obs"], f["var"], f["X"]
        if "perturbation" not in obs:
            raise ValueError(f"Inspect target-gene column before proceeding: {list(obs)}")
        names = strings(obs["perturbation"])
        cell_ids = strings(obs[obs.attrs["_index"]])
        gene_ids = strings(var[var.attrs["_index"]])
        ensembl_ids = strings(var["ensembl_id"]) if "ensembl_id" in var else gene_ids
        if len(set(cell_ids)) != len(cell_ids):
            raise ValueError("Duplicate cell identifiers")
        controls = names == "control"
        if not np.any(controls):
            raise ValueError(f"No explicit control category: {np.unique(names)[:20]}")
        targets, counts = np.unique(names[~controls], return_counts=True)
        rare = set(targets[counts < 2])
        keep = np.asarray([n not in rare for n in names])
        if any("+" in n or "," in n for n in targets):
            raise ValueError("Multi-target perturbations require an explicit grouping rule")
        source_ids = np.flatnonzero(keep)
        targets = np.asarray(sorted(set(names[keep & ~controls])))
        vocabulary = {name: i for i, name in enumerate(targets)}
        labels = np.asarray(
            [-1 if n == "control" else vocabulary[n] for n in names[keep]], dtype=np.int32
        )
        partitions, roles, support = make_splits(labels, seed)
        source_partition = np.full(len(names), -1, dtype=np.int8)
        source_partition[source_ids] = partitions
        gene_sums = np.zeros(len(gene_ids), dtype=np.float64)
        train_count = int(np.sum(partitions == 0))
        for a in range(0, len(names), 4096):
            b = min(a + 4096, len(names))
            block = csr_rows(x, a, b)
            gene_sums += np.asarray(block[source_partition[a:b] == 0].sum(axis=0)).ravel()
        if len(gene_ids) < genes_count:
            raise ValueError("Fewer than requested genes")
        # Deterministic gene-ID tie break, with no validation/test query information.
        selected = np.lexsort((gene_ids, -gene_sums))[:genes_count]
        expression = np.lib.format.open_memmap(
            local / "expression.npy", mode="w+", dtype=np.float32, shape=(len(labels), genes_count)
        )
        output_cursor = 0
        for a in range(0, len(names), 4096):
            b = min(a + 4096, len(names))
            block = csr_rows(x, a, b)[keep[a:b]][:, selected].toarray()
            if not len(block):
                continue
            transformed = normalize_selected(block)
            expression[output_cursor : output_cursor + len(block)] = transformed
            output_cursor += len(block)
        assert output_cursor == len(labels)
        expression.flush()
        symbols = strings(var["gene_name"]) if "gene_name" in var else gene_ids
        csv_write(
            package / "genes.csv",
            [
                {
                    "feature_index": i,
                    "source_gene_index": int(g),
                    "gene_id": ensembl_ids[g],
                    "gene_symbol": symbols[g],
                    "training_mean_raw_counts": gene_sums[g] / train_count,
                }
                for i, g in enumerate(selected)
            ],
        )
    control_mean = np.zeros(genes_count, dtype=np.float64)
    training_controls = np.flatnonzero((labels == -1) & (partitions == 0))
    for a in range(0, len(training_controls), 4096):
        control_mean += expression[training_controls[a : a + 4096]].sum(axis=0, dtype=np.float64)
    control_mean = (control_mean / len(training_controls)).astype(np.float32)
    eval_control = np.empty(len(labels), dtype=np.int64)
    for part in range(3):
        ids = np.flatnonzero(partitions == part)
        pool = np.flatnonzero((partitions == part) & (labels == -1))
        rng = np.random.default_rng(np.random.SeedSequence([seed, 3, part]))
        eval_control[ids] = rng.choice(pool, size=len(ids), replace=True)
    arrays = dict(
        labels=labels,
        partitions=partitions,
        cohorts=roles,
        support=support,
        eval_control=eval_control,
        control_mean=control_mean,
        source_rows=source_ids,
    )
    np.savez_compressed(package / "metadata.npz", **cast(Any, arrays))
    for name, value in arrays.items():
        np.save(local / f"{name}.npy", value, allow_pickle=False)
    csv_write(
        package / "cells.csv.gz",
        [
            {
                "row": i,
                "source_row": int(s),
                "cell_id": cell_ids[s],
                "perturbation": names[s],
                "label_id": int(labels[i]),
                "cell_partition": PARTITIONS[partitions[i]],
                "perturbation_cohort": "control" if labels[i] < 0 else PARTITIONS[roles[i]],
                "support_cell": bool(support[i]),
                "fixed_evaluation_control_row": int(eval_control[i]),
            }
            for i, s in enumerate(source_ids)
        ],
    )
    cohort_rows = []
    for i, name in enumerate(targets):
        mask = labels == i
        cohort_rows.append(
            {
                "label_id": i,
                "perturbation": name,
                "cohort": PARTITIONS[roles[mask][0]],
                "total_cells": int(mask.sum()),
                "training_cells": int(np.sum(mask & (partitions == 0))),
                "validation_cells": int(np.sum(mask & (partitions == 1))),
                "test_cells": int(np.sum(mask & (partitions == 2))),
                "support_cells": int(np.sum(mask & support)),
            }
        )
    csv_write(package / "perturbations.csv", cohort_rows)
    if rare:
        csv_write(
            package / "excluded_perturbations.csv",
            [{"perturbation": name, "reason": "fewer_than_2_cells"} for name in sorted(rare)],
        )
    shards = []
    for a in range(0, len(labels), 4096):
        if shutil.disk_usage(package).free < 600 * 1024**2:
            raise RuntimeError(
                "Less than 600 MiB free on package filesystem; preserve partial output"
            )
        b = min(a + 4096, len(labels))
        path = package / f"expression-{a // 4096:04d}.npz"
        np.savez_compressed(path, x=expression[a:b])
        with np.load(path) as z:
            if not np.array_equal(z["x"], expression[a:b]):
                raise ValueError("Lossless archive round-trip failed")
        assert path.stat().st_size < 50 * 1024**2
        shards.append(
            {
                "file": path.name,
                "row_start": a,
                "row_stop": b,
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
        print(json.dumps({"event": "shard_saved", **shards[-1]}), flush=True)
    info: dict[str, Any] = dict(
        format_version=1,
        source_url=SOURCE_URL,
        source_md5=SOURCE_MD5,
        source_sha256=digest(source),
        source_license="CC-BY-4.0",
        source_doi="10.5281/zenodo.13350497",
        split_seed=seed,
        support_fraction_requested=0.1,
        support_rounding="ceil, at least 1 and at most n-1",
        test_perturbation_fraction=0.2,
        validation_perturbation_fraction=0.1,
        genes=genes_count,
        cells=len(labels),
        perturbations=len(targets),
        gene_selection="highest mean raw count across training-partition cells including training controls/support; gene-ID tie break",
        normalization="select 2000 genes; scale each selected library to 3000; natural log1p; float32; no quantization",
        pearson_delta="per-perturbation Pearson across genes of mean prediction minus training-control mean versus mean target minus same training-control mean; equal-weight macro mean",
        evaluation_controls="fixed unpaired control assignment from disjoint partition-specific control pools",
        expression_sha256=digest(local / "expression.npy"),
        shards=shards,
        elapsed_seconds=time.perf_counter() - started,
    )
    info["files"] = [
        {"file": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(package.iterdir())
        if p.is_file()
    ]
    csv_write(
        package / "preprocessing.csv",
        [
            {"parameter": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
            for k, v in info.items()
            if k not in ("files", "shards")
        ],
    )
    csv_write(package / "files.csv", info["files"])
    (package / "manifest.json").write_text(json.dumps(info, indent=2) + "\n")
    (local / "_SUCCESS").write_text(digest(package / "manifest.json") + "\n")
    print(
        json.dumps(
            {
                "event": "preprocessing_complete",
                "cells": len(labels),
                "perturbations": len(targets),
                "package_bytes": sum(p.stat().st_size for p in package.iterdir() if p.is_file()),
                "seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )


def stage(package: Path, local: Path) -> None:
    started = time.perf_counter()
    info = json.loads((package / "manifest.json").read_text())
    expected = digest(package / "manifest.json")
    local.mkdir(parents=True, exist_ok=True)
    if (local / "_SUCCESS").exists() and (local / "_SUCCESS").read_text().strip() == expected:
        print("Existing verified dataset staging reused", flush=True)
        return
    for item in info["files"]:
        if digest(package / item["file"]) != item["sha256"]:
            raise ValueError(f"Checksum failure: {item['file']}")
    x = np.lib.format.open_memmap(
        local / "expression.npy", mode="w+", dtype=np.float32, shape=(info["cells"], info["genes"])
    )
    for item in info["shards"]:
        with np.load(package / item["file"]) as data:
            x[item["row_start"] : item["row_stop"]] = data["x"]
    x.flush()
    if digest(local / "expression.npy") != info["expression_sha256"]:
        raise ValueError("Staged expression checksum mismatch")
    with np.load(package / "metadata.npz") as z:
        for key in z.files:
            np.save(local / f"{key}.npy", z[key], allow_pickle=False)
    (local / "_SUCCESS").write_text(expected + "\n")
    print(
        json.dumps({"event": "stage_complete", "seconds": time.perf_counter() - started}),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "stage"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--local", required=True, type=Path)
    parser.add_argument("--split-seed", type=int, default=20261021)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.source, args.package, args.local, args.split_seed)
    else:
        stage(args.package, args.local)


if __name__ == "__main__":
    main()
