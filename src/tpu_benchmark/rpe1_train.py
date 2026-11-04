"""TPU port of the frozen RPE1 trainer; model, sampling and metrics are unchanged."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from backend import Backend
from torch import nn
from workload import BASELINE, tensor_hash

from gpu_pilot.rpe1_data import PARTITIONS, csv_write, digest
from gpu_pilot.rpe1_train import (
    CsvLog,
    PerturbationMlp,
    initialize,
    plateau_status,
    save_weights,
    verify_weights,
)


class UnavailableTelemetry:
    """Do not pretend NVIDIA telemetry measures TPU hardware."""

    samples = ()

    def __init__(self, **kwargs):
        pass

    def start(self):
        pass

    def set_phase(self, phase):
        pass

    def stop(self):
        return SimpleNamespace(
            model_dump=lambda: {
                "supported": False,
                "reason": "TPU telemetry not implemented; legacy GPU memory fields are unmeasured zeros",
            }
        )


def train(args: argparse.Namespace) -> None:
    from rpe1_evaluation import evaluate

    if args.phase not in ("final", "smoke", "unit_test"):
        raise ValueError(
            "TPU RPE1 runs use fixed epochs; no hyperparameter resweep or early stopping"
        )
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
    backend = Backend(getattr(args, "backend", "cpu-test"))
    device = backend.device
    torch.set_float32_matmul_precision("highest")
    if backend.name == "xla":
        import torch_xla.backends as xla_backends

        xla_backends.set_mat_mul_precision("highest")
    initial_precision = "highest"
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
    initial_hash = tensor_hash(model.state_dict().items())
    model.to(device)
    if backend.name == "torchtpu":
        model.forward = torch.compile(model.forward, backend="tpu", fullgraph=True)
    backend.sync()
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
        "allocator_memory_limit_gib": None,
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
        "gpu_name": None,
        "accelerator_backend": backend.name,
        "matmul_precision": initial_precision,
        "evaluation_context": "no_grad",
        "initial_state_sha256": initial_hash,
        "runtime": backend.metadata(),
        "baseline_source_sha256": digest(BASELINE / "src/gpu_pilot/rpe1_train.py"),
        "adapter_source_sha256": digest(Path(__file__)),
        "independent_seed_no_ddp": True,
        "precision_qualification": "FP32 tensors with highest TPU matmul precision; not a promise of bitwise CUDA equivalence",
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
    sampler = UnavailableTelemetry(artifact_path=output / "telemetry.jsonl", interval_seconds=2)
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
                batch_stream_hash = hashlib.sha256(order.tobytes() + paired.tobytes()).hexdigest()
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
                    backend.sync()
                    transferred = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(inputs, groups)
                    loss = nn.functional.mse_loss(prediction, targets)
                    loss.backward()  # type: ignore[no-untyped-call]
                    optimizer.step()
                    backend.sync()
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
                "batch_stream_sha256": batch_stream_hash if epoch else "",
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
        "backend": backend.name,
        "runtime": backend.metadata(),
        "initial_state_sha256": initial_hash,
        "matmul_precision": initial_precision,
        "evaluation_context": "no_grad",
        "optimizer_updates": updates,
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
    # Only completion is synchronized; seeds never average gradients or metrics.
    if backend.world_size > 1:
        if backend.name == "xla":
            backend.xm.rendezvous("rpe1-seeds-finished")
        else:
            torch.distributed.barrier()
    backend.close()


def worker(index, args):
    # Global ranks identify independent seeds, never gradient replicas.
    if args.backend == "xla":
        import torch_xla.runtime as xr

        rank = xr.global_ordinal()
    else:
        rank = int(os.environ.get("RANK", "0"))
    args.seed = args.seed_base + rank
    args.output = args.output / f"seed-{args.seed}"
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("xla", "torchtpu", "cpu-test"), required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--local", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("final", "smoke"), required=True)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-eval-cells", type=int, default=0)
    parser.add_argument("--xla-spawn", action="store_true")
    args = parser.parse_args()
    if args.xla_spawn:
        if args.backend != "xla":
            parser.error("--xla-spawn requires XLA")
        import torch_xla

        torch_xla.launch(worker, args=(args,))
    else:
        worker(0, args)
