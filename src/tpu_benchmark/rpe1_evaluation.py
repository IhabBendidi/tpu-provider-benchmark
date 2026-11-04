"""Frozen RPE1 evaluation with an XLA-compatible gradient-disabled context.

PyTorch/XLA 2.9 fails on inference tensors (version_counter error). Use no_grad
on both TPU backends; sampling, aggregation and metrics match the frozen source.
Keep the original GPU trainer unchanged.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn

from gpu_pilot.rpe1_data import PARTITIONS
from gpu_pilot.rpe1_train import CsvLog, pearson_rows, sorted_accumulate


def evaluate(
    model: nn.Module,
    x: np.ndarray,
    labels: np.ndarray,
    partitions: np.ndarray,
    eval_control: np.ndarray,
    control_mean: np.ndarray,
    part: int,
    batch: int,
    device: torch.device,
    perturbations: int,
    per_perturbation_log: CsvLog,
    common: dict[str, Any],
    names: list[str],
    cohorts: list[str],
    max_cells: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    ids = np.flatnonzero((partitions == part) & (labels >= 0))
    ids = ids[np.argsort(labels[ids], kind="stable")]
    if max_cells:
        ids = ids[:max_cells]
    counts = np.bincount(labels[ids], minlength=perturbations)
    predicted = np.zeros((perturbations, x.shape[1]), dtype=np.float64)
    truth = np.zeros_like(predicted)
    squared = np.zeros(perturbations, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for a in range(0, len(ids), batch):
            rows = ids[a : a + batch]
            groups = labels[rows]
            inputs = torch.from_numpy(np.asarray(x[eval_control[rows]])).to(device)
            target_np = np.asarray(x[rows])
            target = torch.from_numpy(target_np).to(device)
            result = model(inputs, torch.from_numpy(groups.astype(np.int64)).to(device))
            per_cell_mse = (
                torch.mean((result - target) ** 2, dim=1).cpu().numpy().astype(np.float64)
            )
            result_np = result.cpu().numpy().astype(np.float64)
            sorted_accumulate(predicted, groups, result_np)
            sorted_accumulate(truth, groups, target_np.astype(np.float64))
            sorted_accumulate(squared, groups, per_cell_mse)
    present = np.flatnonzero(counts)
    prediction_mean = predicted[present] / counts[present, None]
    target_mean = truth[present] / counts[present, None]
    scores = pearson_rows(prediction_mean - control_mean, target_mean - control_mean)
    finite = np.isfinite(scores)
    if not finite.any() or not np.isfinite(squared).all():
        raise RuntimeError("No valid Pearson Delta scores or nonfinite evaluation loss")
    bulk_mse = np.mean((prediction_mean - target_mean) ** 2, axis=1)
    for i, label in enumerate(present):
        per_perturbation_log.row(
            {
                **common,
                "split": PARTITIONS[part],
                "label_id": int(label),
                "perturbation": names[label],
                "perturbation_cohort": cohorts[label],
                "cells": int(counts[label]),
                "mse": squared[label] / counts[label],
                "pseudobulk_mse": bulk_mse[i],
                "pearson_delta": scores[i],
                "pearson_defined": bool(finite[i]),
            }
        )
    return {
        "mse": float(squared.sum() / len(ids)),
        "pearson_delta": float(scores[finite].mean()),
        "pearson_delta_std_across_perturbations": float(scores[finite].std()),
        "pseudobulk_mse": float(bulk_mse.mean()),
        "cells": len(ids),
        "perturbations": len(present),
        "undefined_pearson": int((~finite).sum()),
        "evaluation_seconds": time.perf_counter() - started,
    }
