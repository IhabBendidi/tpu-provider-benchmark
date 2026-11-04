"""TPU execution of the archived AWS protocol, using its original models/loaders.

Warmup and every timed/fixed phase restart model and AdamW. Host loss extraction,
per-step synchronization and three MAX timing reductions match the GPU runner.
Only full runs may be used for historical throughput comparisons.
"""

import argparse
import hashlib
import json
import math
import os
import socket
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from backend import Backend
from workload import (
    BASELINE,
    DistributedContext,
    expression_loader,
    image_loader,
    make_model,
    tensor_hash,
)

from gpu_pilot.config import load_config


def emit(backend, event, **values):
    if backend.rank == 0:
        print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def original_loader(config, backend, validation=False):
    context = DistributedContext(backend.rank, backend.world_size, 0, backend.device)
    return (expression_loader if config.synthetic else image_loader)(config, context, validation)


def validation(model, config, backend, rng_state):
    # The cached GPU loop consumes no CPU random numbers after its first batch.
    # Restore that state so compiler/runtime internals cannot alter validation order.
    torch.set_rng_state(rng_state)
    model.eval()
    numerator = torch.zeros((), device=backend.device)
    denominator = torch.zeros((), device=backend.device)
    count = 0
    with torch.no_grad():
        for batch in original_loader(config, backend, validation=True):
            batch = tuple(value.to(backend.device) for value in batch)
            with backend.autocast(config.run.precision):
                prediction = model.model(batch[0])
                if config.synthetic:
                    error = torch.where(batch[2], (prediction.float() - batch[1]).square(), 0).sum()
                    weight = batch[2].sum()
                else:
                    error = torch.nn.functional.cross_entropy(
                        prediction.float(), batch[1], reduction="sum"
                    )
                    weight = batch[1].numel()
            numerator += error
            denominator += weight
            backend.sync()
            count += 1
    numerator = backend.reduce(numerator)
    denominator = backend.reduce(denominator)
    backend.sync()
    return {
        "loss": float((numerator / denominator).cpu()),
        "denominator": float(denominator.cpu()),
        "batches_per_rank": count,
        "scope": "original_aws_loader",
        "distributed_sampler_may_pad": True,
        "drop_last": bool(config.synthetic),
    }


def run_phase(args, config, backend, phase, seconds=None, updates=None):
    settings = config.synthetic or config.imaging
    accumulation = settings.gradient_accumulation_steps(backend.world_size)
    model = make_model(config)
    with np.load(args.initial_state, allow_pickle=False) as data:
        model.load_state_dict({key: torch.from_numpy(data[key].copy()) for key in data.files})
    variant = getattr(args, "optimization", "baseline")
    if variant in ("sdpa", "combined"):
        from vit_optimizations import apply_attention

        apply_attention(model)
    initial_hash = tensor_hash(model.state_dict().items())
    torch.set_rng_state(torch.from_numpy(np.load(args.initial_state.with_suffix(".rng.npy"))))
    model.to(backend.device).train()
    loader = original_loader(config, backend)
    first = next(iter(loader))
    input_hash = tensor_hash((str(i), value) for i, value in enumerate(first))
    cached = tuple(value.to(backend.device) for value in first)
    validation_rng = torch.get_rng_state()
    function = model
    if backend.name == "torchtpu":
        if variant in ("collective_handshake", "combined"):
            os.environ["TORCH_TPU_SPMD_HANDSHAKE_STAGE"] = "COMPILE_STAGE"
        function = torch.compile(model, backend="tpu", fullgraph=True)
    capture_collectives = variant in ("collective_handshake", "combined")
    if args.communication == "ddp" and not capture_collectives:
        if backend.name != "torchtpu" or backend.world_size == 1:
            raise ValueError("DDP requires distributed TorchTPU")
        function = torch.nn.parallel.DistributedDataParallel(
            function, bucket_cap_mb=getattr(args, "bucket_mb", 25), gradient_as_bucket_view=True
        )
    reduce_gradients = None
    if capture_collectives:

        def average_gradients(gradients):
            outputs = []
            for gradient in gradients:
                value = gradient.clone()
                torch.distributed.all_reduce(value)
                outputs.append(value / backend.world_size)
            return outputs

        reduce_gradients = torch.compile(average_gradients, backend="tpu", fullgraph=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate if config.imaging else 0.001,
        weight_decay=settings.weight_decay if config.imaging else 0.01,
        foreach=False,
        fused=False,
    )
    if variant in ("functional_adamw", "combined"):
        from vit_optimizations import FunctionalAdamW

        optimizer = FunctionalAdamW(
            model.parameters(), settings.learning_rate, settings.weight_decay
        )
    elif variant == "compiled_adamw":
        optimizer.step = torch.compile(optimizer.step, backend="tpu", fullgraph=False)
    backend.sync()
    emit(
        backend,
        "phase_start",
        phase=phase,
        seconds=seconds,
        updates=updates,
        initial_state_sha256=initial_hash,
        first_cpu_batch_sha256=input_hash,
    )
    started = time.perf_counter()
    elapsed, count, last_log = 0.0, 0, 0.0
    step_times, waits = [], []
    first_loss, last_loss = None, None
    while (seconds is not None and elapsed < seconds) or (updates is not None and count < updates):
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        wait, loss_value = 0.0, 0.0
        for micro in range(accumulation):
            data_started = time.perf_counter()
            batch = tuple(value.to(backend.device, non_blocking=True) for value in cached)
            if variant in ("contiguous", "combined"):
                batch = tuple(value.contiguous() for value in batch)
            wait += time.perf_counter() - data_started
            context = (
                function.no_sync()
                if args.communication == "ddp"
                and not capture_collectives
                and micro < accumulation - 1
                else nullcontext()
            )
            with context:
                with backend.autocast(config.run.precision):
                    loss = function(*batch)
                (loss / accumulation).backward()
            backend.microstep()
            loss_value += float(loss.detach().cpu())
        if reduce_gradients is not None:
            parameters = list(model.parameters())
            gradients = reduce_gradients([p.grad for p in parameters])
            for parameter, gradient in zip(parameters, gradients, strict=True):
                parameter.grad = gradient
            optimizer.step()
        else:
            backend.optimizer_step(optimizer, args.communication)
        backend.sync()
        # Preserve all three separate collectives from the historical runner.
        step_times.append(backend.max_seconds(time.perf_counter() - tick))
        waits.append(backend.max_seconds(wait))
        count += 1
        elapsed = backend.max_seconds(time.perf_counter() - started)
        last_loss = loss_value / accumulation
        if not math.isfinite(last_loss):
            raise RuntimeError("Nonfinite training loss")
        if first_loss is None:
            first_loss = last_loss
        if count == 1 or elapsed - last_log >= 5:
            emit(
                backend,
                "progress",
                phase=phase,
                optimizer_updates=count,
                elapsed_seconds=elapsed,
                throughput_samples_per_second=count * settings.global_batch_size / elapsed,
                training_loss=last_loss,
            )
            last_log = elapsed
    validated = validation(model, config, backend, validation_rng)
    if not math.isfinite(validated["loss"]):
        raise RuntimeError("Nonfinite validation loss")
    ordered = sorted(step_times)
    result = {
        "phase": phase,
        "elapsed_seconds": elapsed,
        "optimizer_updates": count,
        "samples_processed": count * settings.global_batch_size,
        "throughput_samples_per_second": count * settings.global_batch_size / elapsed,
        "step_time_median_ms": ordered[int((len(ordered) - 1) * 0.5)] * 1000,
        "step_time_p95_ms": ordered[int((len(ordered) - 1) * 0.95)] * 1000,
        "data_wait_seconds": sum(waits),
        "validation": validated,
        "validation_loss": validated["loss"],
        "first_training_loss": first_loss,
        "last_training_loss": last_loss,
        "initial_state_sha256": initial_hash,
        "first_cpu_batch_sha256": input_hash,
    }
    emit(backend, "phase_complete", **result)
    return result


def run(args):
    torch.set_num_threads(1)
    config = load_config(args.config)
    if config.run.mode != "gpu_only":
        raise ValueError("Historical alignment currently supports cached-input runs only")
    backend = Backend(args.backend)
    try:
        settings = config.synthetic or config.imaging
        if backend.world_size != settings.expected_world_size:
            raise ValueError("World size does not match config")
        output = args.output / f"historical-{args.backend}-rank{backend.rank}.json"
        if output.exists():
            raise FileExistsError(output)
        args.output.mkdir(parents=True, exist_ok=True)
        single_trial = getattr(args, "single_trial", False)
        schedule = [("warmup", 30.0 if single_trial else config.timing.warmup_seconds, None)]
        if getattr(args, "preflight", False):
            schedule = [("preflight", None, 5)]
        schedule += [
            (f"measurement_{i + 1}", config.timing.measurement_seconds, None)
            for i in range(
                0
                if getattr(args, "preflight", False)
                else 1
                if single_trial
                else config.timing.measurement_repetitions
            )
        ]
        if not single_trial and not getattr(args, "preflight", False):
            schedule += [("fixed_job", None, config.timing.fixed_job_updates)]
        record = {
            "schema_version": "tpu-historical-0.1",
            "status": "running",
            "protocol": "aws_original_timing_and_reset",
            "measurement_kind": args.measurement_kind,
            "config": config.model_dump(mode="json"),
            "communication": args.communication,
            "single_trial": single_trial,
            "optimization": getattr(args, "optimization", "baseline"),
            "bucket_mb": getattr(args, "bucket_mb", 25),
            "actual_schedule": schedule,
            "runtime": backend.metadata(),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "baseline_commit": json.loads((BASELINE / "provenance.json").read_text())["git_commit"],
            "adapter_sources": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "phases": [],
            "qualifications": [
                "Archived AWS records have no initial-state/input hashes; bitwise historical identity cannot be verified.",
                "PyTorch/XLA uses torch 2.9.0; TorchTPU and archived AWS use torch 2.14.1.",
                "TPU fixed-shape masked MSE replaces GPU boolean indexing with the same mathematical loss.",
                "AdamW hyperparameters match; TPU foreach=False/fused=False; historical CUDA uses its default selection.",
                "TPU compilation is included wherever the original phase timer includes the first step.",
                "No checkpoint/export or GPU telemetry overhead comparison; training throughput and fixed-update validation only.",
            ],
        }
        for phase, seconds, updates in schedule:
            record["phases"].append(run_phase(args, config, backend, phase, seconds, updates))
            temporary = output.with_suffix(".partial")
            temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        record["status"] = "completed"
        record["runtime"] = backend.metadata()
        output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        output.with_suffix(".partial").unlink(missing_ok=True)
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("xla", "torchtpu", "cpu-test"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--communication", choices=("per-parameter", "ddp"), default="per-parameter"
    )
    parser.add_argument("--measurement-kind", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--single-trial", action="store_true")
    parser.add_argument("--bucket-mb", type=int, default=25)
    parser.add_argument(
        "--optimization",
        default="baseline",
        choices=(
            "baseline",
            "compiled_adamw",
            "functional_adamw",
            "collective_handshake",
            "sdpa",
            "contiguous",
            "combined",
        ),
    )
    parser.add_argument("--xla-spawn", action="store_true")
    args = parser.parse_args()
    if args.xla_spawn:
        if args.backend != "xla":
            parser.error("--xla-spawn requires XLA")
        import torch_xla

        torch_xla.launch(worker, args=(args,))
    else:
        run(args)


def worker(index, args):
    run(args)


if __name__ == "__main__":
    main()
