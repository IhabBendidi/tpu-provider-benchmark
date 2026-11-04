"""Matched PyTorch/XLA and TorchTPU training, separate from the GPU runner.

Start with a bounded smoke run; --steps/--warmup-steps/--repetitions control
measurement. All updates, including warmup, are recorded. No implicit CPU mode.
"""

import argparse
import hashlib
import json
import math
import os
import socket
import statistics
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from backend import Backend
from workload import BASELINE, make_loader, make_model, tensor_hash, validate

from gpu_pilot.config import load_config


def positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def emit(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def run(args):
    torch.set_num_threads(1)
    started_at_utc = datetime.now(UTC).isoformat()
    config = load_config(args.config)
    backend = Backend(args.backend)
    try:
        settings = config.synthetic or config.imaging
        if settings.expected_world_size != backend.world_size:
            raise ValueError(
                f"Expected {settings.expected_world_size} devices, got {backend.world_size}"
            )
        accumulation = settings.gradient_accumulation_steps(backend.world_size)
        if accumulation < 1:
            raise ValueError("Global batch must be at least one microbatch per rank")
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        result_path = output / f"{args.backend}-rank{backend.rank}.json"
        if result_path.exists():
            raise FileExistsError(f"Refusing to overwrite {result_path}")
        started = time.perf_counter()
        model = make_model(config)
        if args.initial_state:
            with np.load(args.initial_state, allow_pickle=False) as data:
                model.load_state_dict(
                    {name: torch.from_numpy(data[name].copy()) for name in data.files}, strict=True
                )
        initial_hash = tensor_hash(model.state_dict().items())
        model.to(backend.device)
        model.train()
        function = model
        if args.backend == "torchtpu" and not args.eager:
            function = torch.compile(model, backend="tpu", fullgraph=True)
        if args.communication == "ddp":
            if args.backend != "torchtpu" or backend.world_size == 1:
                raise ValueError("DDP experiment requires distributed TorchTPU")
            function = torch.nn.parallel.DistributedDataParallel(
                function, bucket_cap_mb=25, gradient_as_bucket_view=True
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate if config.imaging else 1e-3,
            weight_decay=settings.weight_decay if config.imaging else 0.01,
            foreach=False,
            fused=False,
        )
        loader = make_loader(config, backend)
        if len(loader) == 0:
            raise ValueError("Training shard is smaller than one microbatch")
        iterator = iter(loader)
        first_cpu = next(iterator)
        input_hash = tensor_hash((str(i), value) for i, value in enumerate(first_cpu))
        cached = tuple(value.to(backend.device) for value in first_cpu)
        epoch = 0
        # Restart so cached and pipeline modes begin from the same first batch.
        iterator = iter(loader)

        def next_batch():
            nonlocal iterator, epoch
            if config.run.mode == "gpu_only":
                return cached
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                loader.sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            return tuple(value.to(backend.device) for value in batch)

        def update():
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for micro in range(accumulation):
                batch = next_batch()
                context = (
                    function.no_sync()
                    if args.communication == "ddp" and micro < accumulation - 1
                    else nullcontext()
                )
                with context:
                    with backend.autocast(config.run.precision):
                        loss = function(*batch)
                    (loss / accumulation).backward()
                losses.append(loss.detach())
                backend.microstep()
            backend.optimizer_step(optimizer, args.communication)
            # Include queued device execution, not merely Python dispatch time.
            average = torch.stack(losses).mean()
            backend.sync()
            return average

        backend.sync()
        setup_seconds = time.perf_counter() - started
        warmup_started = time.perf_counter()
        warmup_losses = []
        for step in range(args.warmup_steps):
            tick = time.perf_counter()
            loss = float(update().cpu())
            if not math.isfinite(loss):
                raise RuntimeError("Nonfinite warmup loss")
            warmup_losses.append(loss)
            emit(
                "warmup",
                backend=args.backend,
                rank=backend.rank,
                update=step + 1,
                seconds=time.perf_counter() - tick,
                loss=loss,
            )
        warmup_seconds = backend.max_seconds(time.perf_counter() - warmup_started)
        trials = []
        for repetition in range(args.repetitions):
            backend.sync()
            # Reduce/barrier before starting each timed window.
            backend.max_seconds(0.0)
            trial_started = time.perf_counter()
            step_times, losses = [], []
            for update_index in range(args.steps):
                tick = time.perf_counter()
                loss = update()
                step_times.append(time.perf_counter() - tick)
                losses.append(loss)
                if backend.rank == 0 and (update_index + 1) % 100 == 0:
                    emit(
                        "progress",
                        backend=args.backend,
                        repetition=repetition + 1,
                        updates=update_index + 1,
                        elapsed_seconds=time.perf_counter() - trial_started,
                    )
            # Do host loss extraction after the timing window on both backends.
            elapsed = backend.max_seconds(time.perf_counter() - trial_started)
            host_losses = [float(value.cpu()) for value in losses]
            if not all(math.isfinite(value) for value in host_losses):
                raise RuntimeError("Nonfinite measured loss")
            trial = {
                "elapsed_seconds": elapsed,
                "optimizer_updates": args.steps,
                "samples_processed": args.steps * settings.global_batch_size,
                "throughput_samples_per_second": args.steps * settings.global_batch_size / elapsed,
                "local_step_median_ms": statistics.median(step_times) * 1000,
                "local_step_p95_ms": sorted(step_times)[math.ceil(0.95 * len(step_times)) - 1]
                * 1000,
                "losses": host_losses,
            }
            trials.append(trial)
            emit(
                "trial", backend=args.backend, rank=backend.rank, repetition=repetition + 1, **trial
            )
        validation = validate(model, config, backend, args.validation_batches)
        if not math.isfinite(validation["loss"]):
            raise RuntimeError("Nonfinite validation loss")
        record = {
            "schema_version": "tpu-comparison-0.1",
            "measurement_kind": args.measurement_kind,
            "communication": args.communication,
            "status": "completed",
            "started_at_utc": started_at_utc,
            "runtime": backend.metadata(),
            "hostname": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "execution": "torch_compile_model_and_loss"
            if args.backend == "torchtpu" and not args.eager
            else "xla_lazy"
            if args.backend == "xla"
            else "eager",
            "baseline_commit": json.loads((BASELINE / "provenance.json").read_text())["git_commit"],
            "adapter_sources": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "initialization_source": str(args.initial_state)
            if args.initial_state
            else "backend_cpu_rng",
            "config_sha256": hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            "config": config.model_dump(mode="json"),
            "initial_state_sha256": initial_hash,
            "first_cpu_batch_sha256": input_hash,
            "gradient_accumulation_steps": accumulation,
            "setup_seconds": setup_seconds,
            "warmup_seconds": warmup_seconds,
            "warmup_updates": args.warmup_steps,
            "warmup_losses": warmup_losses,
            "trials": trials,
            "validation": validation,
            "total_optimizer_updates": args.warmup_steps + args.steps * args.repetitions,
            "peak_accelerator_memory_bytes": None,
            "diagnostics": [
                "Warmup includes initial compilation; trials continue the warmed model.",
                "Optimizer is unfused AdamW with foreach=False on both backends.",
                "No device utilization or peak memory measurement implemented yet.",
                "Historical GPU trials reset model state; this protocol must be compared separately.",
            ],
        }
        result_path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        emit("completed", path=str(result_path))
    finally:
        backend.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("xla", "torchtpu", "cpu-test"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-kind", choices=("smoke", "pilot"), default="smoke")
    parser.add_argument("--initial-state", type=Path, help="Shared CPU initialization .npz fixture")
    parser.add_argument("--steps", type=positive, default=10)
    parser.add_argument("--warmup-steps", type=positive, default=5)
    parser.add_argument("--repetitions", type=positive, default=3)
    parser.add_argument("--validation-batches", type=positive)
    parser.add_argument("--eager", action="store_true", help="Optional TorchTPU diagnostic only")
    parser.add_argument("--communication", choices=("per-parameter", "ddp"), default="per-parameter")
    parser.add_argument("--xla-spawn", action="store_true", help="Launch all local XLA TPU devices")
    args = parser.parse_args()
    if args.xla_spawn:
        if args.backend != "xla":
            parser.error("--xla-spawn requires --backend xla")
        os.environ["PJRT_DEVICE"] = "TPU"
        import torch_xla

        torch_xla.launch(_xla_worker, args=(args,))
    else:
        run(args)


def _xla_worker(index, args):
    run(args)


if __name__ == "__main__":
    main()
