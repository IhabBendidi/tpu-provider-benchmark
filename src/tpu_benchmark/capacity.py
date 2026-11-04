"""One fresh process per ViT capacity candidate; no activation checkpointing."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from backend import Backend
from workload import make_model

from gpu_pilot.config import load_config


def memory(backend):
    try:
        if backend.name == "xla":
            return backend.xm.get_memory_info(str(backend.device))
        free, total = torch.accelerator.get_memory_info()
        return {
            "free_bytes": free,
            "total_bytes": total,
            "peak_allocated_bytes": torch.accelerator.max_memory_allocated(),
            "peak_reserved_bytes": torch.accelerator.max_memory_reserved(),
        }
    except Exception as error:
        return {"memory_statistics_unavailable": str(error)}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    config = load_config(args.config)
    backend = Backend(args.backend)
    if backend.world_size != 1:
        raise ValueError("Capacity candidates run on one chip")
    record = {
        "schema_version": "tpu-capacity-0.1",
        "status": "running",
        "batch_per_chip": args.batch,
        "gradient_accumulation_steps": 1,
        "precision": "bf16",
        "optimizer": "AdamW; FP32 parameters and moments",
        "activation_checkpointing": False,
        "optimizer_updates": 0,
        "config": config.model_dump(mode="json"),
        "runtime": backend.metadata(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "memory_before": memory(backend),
        "qualification": "Single-chip fit test; distributed DDP can require additional memory. Not a throughput comparison or maximum-batch search.",
    }
    started = time.perf_counter()
    code = 0
    try:
        model = make_model(config).to(backend.device).train()
        function = (
            torch.compile(model, backend="tpu", fullgraph=True)
            if backend.name == "torchtpu"
            else model
        )
        settings = config.imaging
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate,
            weight_decay=settings.weight_decay,
            foreach=False,
            fused=False,
        )
        generator = torch.Generator().manual_seed(config.run.seed)
        images = torch.randn(
            args.batch, 3, settings.image_size, settings.image_size, generator=generator
        ).to(backend.device)
        labels = (torch.arange(args.batch) % settings.class_count).to(backend.device)
        backend.sync()
        if backend.name == "torchtpu":
            torch.accelerator.reset_peak_memory_stats()
        for step in range(5):
            optimizer.zero_grad(set_to_none=True)
            with backend.autocast("bf16"):
                loss = function(images, labels)
            loss.backward()
            backend.microstep()
            backend.optimizer_step(optimizer)
            backend.sync()
            record["loss"] = float(loss.cpu())
            if not math.isfinite(record["loss"]):
                raise RuntimeError("Nonfinite capacity-test loss")
            record["optimizer_updates"] = step + 1
            print(
                json.dumps(
                    {
                        "event": "capacity_progress",
                        "batch_per_chip": args.batch,
                        "optimizer_updates": step + 1,
                    }
                ),
                flush=True,
            )
        record["status"] = "fits_five_updates"
        record["memory_after"] = memory(backend)
    except Exception as error:
        diagnostic = str(error)
        # Preserve unknown compiler/runtime failures separately from memory exhaustion.
        exhausted = isinstance(error, torch.OutOfMemoryError) or any(
            text in diagnostic.lower()
            for text in ("out of memory", "resource_exhausted", "ran out of memory")
        )
        record["status"] = "memory_or_resource_exhausted" if exhausted else "unexpected_error"
        record["diagnostic"] = diagnostic
        record["error_type"] = type(error).__name__
        code = 42 if exhausted else 1
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        backend.close()
    return code


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("xla", "torchtpu", "cpu-test"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("Batch must be positive")
    raise SystemExit(run(args))
