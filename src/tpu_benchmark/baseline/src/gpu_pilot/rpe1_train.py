"""Single-device, fixed-batch convergence benchmark with portable initialization."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from gpu_pilot.rpe1_data import PARTITIONS, csv_write, digest
from gpu_pilot.telemetry import NvidiaSmiSampler


class PerturbationMlp(nn.Module):
    def __init__(self, genes: int, perturbations: int, hidden: int = 256, latent: int = 128):
        super().__init__()
        self.perturbations = perturbations
        self.encoder = nn.Sequential(
            nn.Linear(genes, hidden), nn.ReLU(), nn.Linear(hidden, latent), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent + perturbations, hidden), nn.ReLU(), nn.Linear(hidden, genes)
        )

    def forward(self, controls: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        identity = nn.functional.one_hot(labels, self.perturbations).to(controls.dtype)
        return cast(
            torch.Tensor, self.decoder(torch.cat((self.encoder(controls), identity), dim=1))
        )


def initialize(model: nn.Module, seed: int) -> None:
    """NumPy PCG64 uniform init; exported arrays avoid framework PRNG differences."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 10]))
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                bound = 1 / math.sqrt(layer.in_features)
                for parameter in (layer.weight, layer.bias):
                    values = rng.uniform(-bound, bound, tuple(parameter.shape)).astype(np.float32)
                    parameter.copy_(torch.from_numpy(values))


def pearson_rows(prediction_delta: np.ndarray, target_delta: np.ndarray) -> np.ndarray:
    a = prediction_delta.astype(np.float64)
    b = target_delta.astype(np.float64)
    a -= a.mean(axis=1, keepdims=True)
    b -= b.mean(axis=1, keepdims=True)
    denominator = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    result = np.full(len(a), np.nan)
    np.divide(np.sum(a * b, axis=1), denominator, out=result, where=denominator > 1e-12)
    return np.asarray(np.clip(result, -1, 1))


def save_weights(path: Path, model: nn.Module) -> None:
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(
        temporary, **{k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    )
    temporary.replace(path)


def plateau_status(
    records: list[dict[str, Any]],
    minimum_epochs: int = 100,
    window: int = 10,
    patience: int = 40,
    mse_delta: float = 1e-5,
    pearson_delta: float = 0.001,
) -> dict[str, Any]:
    """Require both validation metrics to stop improving over complete blocks.

    Anchors advance only on a meaningful improvement, so accumulated small
    improvements count. Test and training metrics never enter the decision.
    """
    if window < 1 or patience < window or minimum_epochs < 2 * window:
        raise ValueError("Invalid validation plateau bounds")
    mse_anchor, pearson_anchor = math.inf, -math.inf
    mse_epoch = pearson_epoch = 0
    completed = int(records[-1]["epoch"]) if records else 0
    by_epoch = {int(r["epoch"]): r for r in records}
    latest_mse = latest_pearson = math.nan
    for end in range(window, completed + 1, window):
        block = [by_epoch[e] for e in range(end - window + 1, end + 1)]
        latest_mse = float(np.mean([float(r["validation_mse"]) for r in block]))
        latest_pearson = float(np.mean([float(r["validation_pearson_delta"]) for r in block]))
        if not math.isfinite(latest_mse) or not math.isfinite(latest_pearson):
            raise ValueError("Nonfinite validation plateau metrics")
        if latest_mse < mse_anchor - mse_delta:
            mse_anchor, mse_epoch = latest_mse, end
        if latest_pearson > pearson_anchor + pearson_delta:
            pearson_anchor, pearson_epoch = latest_pearson, end
    last = max(mse_epoch, pearson_epoch)
    return {
        "plateau": completed >= minimum_epochs
        and completed % window == 0
        and completed - last >= patience,
        "epoch": completed,
        "window_epochs": window,
        "window_validation_mse": latest_mse,
        "window_validation_pearson_delta": latest_pearson,
        "mse_anchor": mse_anchor,
        "pearson_anchor": pearson_anchor,
        "last_mse_improvement_epoch": mse_epoch,
        "last_pearson_improvement_epoch": pearson_epoch,
        "epochs_without_material_improvement": completed - last,
    }


def verify_weights(path: Path, model: nn.Module) -> list[dict[str, Any]]:
    """Confirm that deterministic replay reproduces the original trajectory."""
    rows = []
    with np.load(path) as reference:
        if set(reference.files) != set(model.state_dict()):
            raise ValueError("Replay model structure differs")
        for key, tensor in model.state_dict().items():
            current = tensor.detach().cpu().numpy()
            rows.append(
                {
                    "parameter": key,
                    "exact_equal": np.array_equal(current, reference[key]),
                    "maximum_absolute_difference": float(np.max(np.abs(current - reference[key]))),
                }
            )
    return rows


class CsvLog:
    def __init__(self, path: Path):
        self.path = path
        self.handle = (
            gzip.open(path, "wt", newline="")
            if path.suffix == ".gz"
            else path.open("w", newline="")
        )
        self.writer: csv.DictWriter[str] | None = None

    def row(self, data: dict[str, Any]) -> None:
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(data))
            self.writer.writeheader()
        self.writer.writerow(data)

    def flush(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self) -> None:
        self.flush()
        self.handle.close()


def sorted_accumulate(output: np.ndarray, labels: np.ndarray, values: np.ndarray) -> None:
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    output[labels[starts]] += np.add.reduceat(values, starts, axis=0)


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
    with torch.inference_mode():
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


def train(args: argparse.Namespace) -> None:
    setup_started = time.perf_counter()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob("epochs.csv")):
        raise ValueError("Refusing to overwrite a previous run")
    info = json.loads((args.package / "manifest.json").read_text())
    data_hash = digest(args.package / "manifest.json")
    if (args.local / "_SUCCESS").read_text().strip() != data_hash:
        raise ValueError("Staged dataset and portable package differ")
    saturation = args.phase == "saturation"
    if args.phase == "final" or saturation:
        if args.epochs < 1 or args.max_train_steps or args.max_eval_cells:
            raise ValueError("Final runs require all cells and a positive epoch count")
        selection = json.loads(args.selection.read_text())
        if selection["data_manifest_sha256"] != data_hash:
            raise ValueError("Hyperparameter selection used different data")
        args.learning_rate = selection["learning_rate"]
    if saturation:
        if args.epochs < args.minimum_epochs or args.minimum_epochs < 30:
            raise ValueError("Saturation runs must replay all 30 original epochs")
        baseline = json.loads((args.baseline_run / "configuration.json").read_text())
        for key, value in {
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "data_manifest_sha256": data_hash,
            "batch_size": args.batch_size,
        }.items():
            if baseline[key] != value:
                raise ValueError(f"Replay baseline mismatch: {key}")
    if args.batch_size != 512 and args.phase != "unit_test":
        raise ValueError("The GPU/TPU comparison fixes the batch size at 512")
    device = torch.device(args.device)
    if device.type == "cuda":
        if torch.cuda.device_count() != 1:
            raise ValueError("Each job must reserve and expose exactly one GPU")
        torch.cuda.set_device(0)
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, 4 * 1024**3 / total), 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    model = PerturbationMlp(info["genes"], info["perturbations"])
    initialize(model, args.seed)
    if saturation:
        verification = verify_weights(args.baseline_run / "initial_weights.npz", model)
        if not all(row["exact_equal"] for row in verification):
            raise ValueError("Initial weights differ from original run")
        os.link(args.baseline_run / "initial_weights.npz", output / "initial_weights.npz")
    if args.phase == "final":
        save_weights(output / "initial_weights.npz", model)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        foreach=False,
        fused=False,
    )
    x = np.load(args.local / "expression.npy", mmap_mode="r")
    labels = np.load(args.local / "labels.npy")
    partitions = np.load(args.local / "partitions.npy")
    eval_control = np.load(args.local / "eval_control.npy")
    control_mean = np.load(args.local / "control_mean.npy")
    train_rows = np.flatnonzero((partitions == 0) & (labels >= 0))
    controls = np.flatnonzero((partitions == 0) & (labels == -1))
    perturbation_rows = list(csv.DictReader((args.package / "perturbations.csv").open()))
    names = [row["perturbation"] for row in perturbation_rows]
    cohorts = [row["cohort"] for row in perturbation_rows]
    common = {
        "run_id": f"{output.parent.name}/{output.name}",
        "phase": args.phase,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
    }
    configuration = {
        **common,
        "epochs": args.epochs,
        "split_seed": info["split_seed"],
        "data_manifest_sha256": data_hash,
        "features": info["genes"],
        "perturbations": info["perturbations"],
        "encoder_sizes": [info["genes"], 256, 128],
        "decoder_sizes": [128 + info["perturbations"], 256, info["genes"]],
        "conditioning": "concatenate encoder embedding and full perturbation one-hot vector",
        "activation": "ReLU",
        "output_activation": "linear",
        "dropout": 0.0,
        "pretrained": False,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "precision": "float32",
        "tf32": False,
        "deterministic_algorithms": True,
        "allocator_memory_limit_gib": 4,
        "optimizer": "AdamW",
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 1e-4,
        "optimizer_foreach": False,
        "optimizer_fused": False,
        "scheduler": "none",
        "gradient_accumulation": 1,
        "gradient_clipping": "none",
        "loss": "MSE averaged over cells and all selected genes",
        "initialization": "NumPy PCG64 SeedSequence([seed,10]); independent uniform(-1/sqrt(fan_in),1/sqrt(fan_in)) for each Linear weight then bias",
        "training_order": "NumPy PCG64 SeedSequence([seed,20,epoch]); permutation followed by choice of training control rows with replacement",
        "control_pairing": "unpaired random controls during training; fixed per-cell controls for evaluation",
        "test_used_for_selection": False,
        "selection_rule": "maximum validation macro Pearson Delta; break tie by lower validation MSE",
        "epochs_include_epoch_zero": True,
        "max_train_steps": args.max_train_steps,
        "max_eval_cells": args.max_eval_cells,
        "checkpoint_format": "NPZ mapping PyTorch state_dict names to float32 arrays; linear weights are [out,in]",
        "host": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "training_rows": len(train_rows),
        "setup_seconds": time.perf_counter() - setup_started,
    }
    if saturation:
        configuration.update(
            {
                "baseline_run": str(args.baseline_run),
                "replay_epoch": 30,
                "minimum_epochs": args.minimum_epochs,
                "maximum_epochs": args.epochs,
                "plateau_window": args.plateau_window,
                "plateau_patience": args.plateau_patience,
                "plateau_mse_delta": args.plateau_mse_delta,
                "plateau_pearson_delta": args.plateau_pearson_delta,
                "stopping_rule": "both validation block means have no material improvement for patience epochs; minimum epoch bound applies",
                "resume_checkpoint": str(args.checkpoint_path),
            }
        )
    csv_write(
        output / "configuration.csv",
        [
            {"parameter": k, "value": json.dumps(v) if isinstance(v, (dict, list)) else v}
            for k, v in configuration.items()
        ],
    )
    (output / "configuration.json").write_text(json.dumps(configuration, indent=2) + "\n")
    sampler = NvidiaSmiSampler(artifact_path=output / "telemetry.jsonl", interval_seconds=2)
    sampler.start()
    epochs, steps, per_pert = (
        CsvLog(output / "epochs.csv"),
        CsvLog(output / ("steps.csv.gz" if saturation else "steps.csv")),
        CsvLog(output / "perturbations.csv.gz"),
    )
    epoch_records = []
    updates, cumulative_train, best_score, best_mse, best_epoch = 0, 0.0, -math.inf, math.inf, -1
    run_start = time.perf_counter()
    stop_reason = "maximum_epochs" if saturation else "fixed_epochs"
    plateau_records = []
    try:
        for epoch in range(args.epochs + 1):
            epoch_start = time.perf_counter()
            train_seconds, online_sum, online_count = 0.0, 0.0, 0
            if epoch:
                sampler.set_phase(f"epoch{epoch}-train")
                model.train()
                rng = np.random.default_rng(np.random.SeedSequence([args.seed, 20, epoch]))
                order = rng.permutation(train_rows)
                paired = rng.choice(controls, size=len(order), replace=True)
                train_start = time.perf_counter()
                for batch_index, a in enumerate(range(0, len(order), args.batch_size)):
                    if args.max_train_steps and batch_index >= args.max_train_steps:
                        break
                    tick = time.perf_counter()
                    rows = order[a : a + args.batch_size]
                    input_cpu = torch.from_numpy(np.asarray(x[paired[a : a + len(rows)]]))
                    target_cpu = torch.from_numpy(np.asarray(x[rows]))
                    group_cpu = torch.from_numpy(labels[rows].astype(np.int64))
                    loaded = time.perf_counter()
                    inputs, targets, groups = (
                        input_cpu.to(device),
                        target_cpu.to(device),
                        group_cpu.to(device),
                    )
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    transferred = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(inputs, groups)
                    loss = nn.functional.mse_loss(prediction, targets)
                    loss.backward()  # type: ignore[no-untyped-call]
                    optimizer.step()
                    loss_value = float(loss.detach().cpu())
                    if not math.isfinite(loss_value):
                        raise RuntimeError("Nonfinite training loss")
                    done = time.perf_counter()
                    updates += 1
                    online_sum += loss_value * len(rows)
                    online_count += len(rows)
                    steps.row(
                        {
                            **common,
                            "epoch": epoch,
                            "update": updates,
                            "batch_index": batch_index,
                            "cells": len(rows),
                            "mse": loss_value,
                            "read_seconds": loaded - tick,
                            "h2d_seconds": transferred - loaded,
                            "compute_seconds": done - transferred,
                            "step_seconds": done - tick,
                            "wall_seconds": done - run_start,
                        }
                    )
                    if batch_index % 100 == 0:
                        steps.flush()
                        print(
                            json.dumps(
                                {
                                    "event": "training_progress",
                                    **common,
                                    "epoch": epoch,
                                    "update": updates,
                                    "mse": loss_value,
                                }
                            ),
                            flush=True,
                        )
                train_seconds = time.perf_counter() - train_start
                cumulative_train += train_seconds
            values = {}
            for part in (
                (0, 1, 2) if args.phase in ("final", "saturation", "smoke", "unit_test") else (0, 1)
            ):
                sampler.set_phase(f"epoch{epoch}-{PARTITIONS[part]}")
                metrics = evaluate(
                    model,
                    x,
                    labels,
                    partitions,
                    eval_control,
                    control_mean,
                    part,
                    args.batch_size,
                    device,
                    info["perturbations"],
                    per_pert,
                    {**common, "epoch": epoch},
                    names,
                    cohorts,
                    args.max_eval_cells,
                )
                values.update({f"{PARTITIONS[part]}_{k}": v for k, v in metrics.items()})
            score, mse = values["validation_pearson_delta"], values["validation_mse"]
            improved = score > best_score or (score == best_score and mse < best_mse)
            if improved:
                best_score, best_mse, best_epoch = score, mse, epoch
                if args.phase == "final" or saturation:
                    save_weights(output / "best_weights.npz", model)
            row = {
                **common,
                "epoch": epoch,
                "optimizer_updates": updates,
                "online_training_mse": online_sum / online_count if online_count else "",
                **values,
                "train_seconds": train_seconds,
                "cumulative_train_seconds": cumulative_train,
                "epoch_wall_seconds": time.perf_counter() - epoch_start,
                "wall_seconds": time.perf_counter() - run_start,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated()
                if device.type == "cuda"
                else 0,
                "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved()
                if device.type == "cuda"
                else 0,
                "best_validation_epoch_so_far": best_epoch,
                "selected_checkpoint_updated": improved,
            }
            epochs.row(row)
            epoch_records.append(row)
            for log in (epochs, steps, per_pert):
                log.flush()
            print(json.dumps({"event": "epoch_complete", **row}), flush=True)
            if saturation:
                if epoch == 30:
                    verification = verify_weights(args.baseline_run / "final_weights.npz", model)
                    csv_write(output / "replay_verification.csv", verification)
                    if not all(r["exact_equal"] for r in verification):
                        raise RuntimeError("Epoch-30 replay does not match the original weights")
                status = plateau_status(
                    epoch_records,
                    args.minimum_epochs,
                    args.plateau_window,
                    args.plateau_patience,
                    args.plateau_mse_delta,
                    args.plateau_pearson_delta,
                )
                if epoch and epoch % args.plateau_window == 0:
                    plateau_records.append(status)
                    csv_write(output / "plateau.csv", plateau_records)
                    print(json.dumps({"event": "plateau_check", **common, **status}), flush=True)
                if status["plateau"]:
                    stop_reason = "validation_plateau"
                if epoch and (epoch % 10 == 0 or status["plateau"] or epoch == args.epochs):
                    args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = args.checkpoint_path.with_suffix(".partial.pt")
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "epoch_records": epoch_records,
                            "updates": updates,
                            "cumulative_train_seconds": cumulative_train,
                            "best_score": best_score,
                            "best_mse": best_mse,
                            "best_epoch": best_epoch,
                            "configuration": configuration,
                            "torch_rng_state": torch.get_rng_state(),
                            "cuda_rng_state": torch.cuda.get_rng_state()
                            if device.type == "cuda"
                            else None,
                        },
                        temporary,
                    )
                    temporary.replace(args.checkpoint_path)
                if status["plateau"]:
                    break
        if args.phase == "final" or saturation:
            save_weights(output / "final_weights.npz", model)
        convergence = []
        for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            reached = next(
                (r for r in epoch_records if r["validation_pearson_delta"] >= threshold), None
            )
            convergence.append(
                {
                    **common,
                    "metric": "validation_pearson_delta",
                    "threshold": threshold,
                    "reached": reached is not None,
                    "epoch": reached["epoch"] if reached else "",
                    "updates": reached["optimizer_updates"] if reached else "",
                    "train_seconds": reached["cumulative_train_seconds"] if reached else "",
                    "wall_seconds": reached["wall_seconds"] if reached else "",
                }
            )
        csv_write(output / "convergence.csv", convergence)
        for threshold in (1.0, 0.75, 0.5, 0.25, 0.1, 0.05):
            reached = next((r for r in epoch_records if r["validation_mse"] <= threshold), None)
            convergence.append(
                {
                    **common,
                    "metric": "validation_mse",
                    "threshold": threshold,
                    "reached": reached is not None,
                    "epoch": reached["epoch"] if reached else "",
                    "updates": reached["optimizer_updates"] if reached else "",
                    "train_seconds": reached["cumulative_train_seconds"] if reached else "",
                    "wall_seconds": reached["wall_seconds"] if reached else "",
                }
            )
        csv_write(output / "convergence.csv", convergence)
    finally:
        telemetry = sampler.stop()
        for log in (epochs, steps, per_pert):
            log.close()
        telemetry_rows = []
        for sample in sampler.samples:
            for gpu in sample["gpus"]:
                telemetry_rows.append(
                    {
                        "timestamp_utc": sample["timestamp_utc"],
                        "phase": sample["phase"],
                        "hostname": sample["hostname"],
                        **gpu,
                        "host_cpu_percent": sample["host"]["cpu_utilization_percent"],
                        "host_memory_used_mib": sample["host"]["memory_used_mib"],
                    }
                )
        if telemetry_rows:
            csv_write(output / "telemetry.csv", telemetry_rows)
        csv_write(
            output / "telemetry_summary.csv",
            [{"parameter": k, "value": v} for k, v in telemetry.model_dump().items()],
        )
    if len(epoch_records) != args.epochs + 1 and stop_reason != "validation_plateau":
        raise RuntimeError("Incomplete epoch history")
    summary = {
        **common,
        "status": "complete",
        "epochs": int(epoch_records[-1]["epoch"]),
        "maximum_epochs": args.epochs,
        "stop_reason": stop_reason,
        "data_manifest_sha256": data_hash,
        "best_validation_epoch": best_epoch,
        "best_validation_pearson_delta": best_score,
        "best_validation_mse": best_mse,
        "total_seconds": time.perf_counter() - setup_started,
        "training_seconds": cumulative_train,
        "parameter_count": configuration["parameter_count"],
        "final_train_mse": epoch_records[-1]["train_mse"],
        "final_validation_mse": epoch_records[-1]["validation_mse"],
        "final_validation_pearson_delta": epoch_records[-1]["validation_pearson_delta"],
        "final_test_mse": epoch_records[-1].get("test_mse", ""),
        "final_test_pearson_delta": epoch_records[-1].get("test_pearson_delta", ""),
        "best_checkpoint_test_pearson_delta": epoch_records[best_epoch].get(
            "test_pearson_delta", ""
        ),
        "peak_gpu_allocated_bytes": epoch_records[-1]["peak_gpu_allocated_bytes"],
        "peak_gpu_reserved_bytes": epoch_records[-1]["peak_gpu_reserved_bytes"],
    }
    csv_write(output / "summary.csv", [summary])
    files = [
        {"file": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
        for p in sorted(output.iterdir())
        if p.is_file()
    ]
    csv_write(output / "files.csv", files)
    (output / "_SUCCESS").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"event": "run_complete", **summary}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase", choices=("sweep", "final", "saturation", "smoke", "unit_test"), required=True
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-cells", type=int, default=0)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--minimum-epochs", type=int, default=100)
    parser.add_argument("--plateau-window", type=int, default=10)
    parser.add_argument("--plateau-patience", type=int, default=40)
    parser.add_argument("--plateau-mse-delta", type=float, default=1e-5)
    parser.add_argument("--plateau-pearson-delta", type=float, default=0.001)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
