"""Diagnostic phase timing and native traces; never an archived H100 ranking.

Normal updates preserve overlap. Segmented updates fence stages intentionally;
their timings describe serialized stages, not additive costs of the normal run.
"""

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
from backend import Backend
from historical_benchmark import original_loader
from workload import make_model, tensor_hash

from gpu_pilot.config import load_config


@contextmanager
def stage(name, backend, timings, segmented, tracing):
    if segmented:
        backend.sync()
    if tracing and backend.name == "xla":
        import torch_xla.debug.profiler as xp

        scope = xp.Trace(name)
    elif tracing:
        scope = torch.profiler.record_function(name)
    else:
        scope = nullcontext()
    started = time.perf_counter()
    with scope:
        yield
    if segmented:
        backend.sync()
        timings[name] += time.perf_counter() - started


def update(
    function,
    model,
    optimizer,
    batch,
    backend,
    accumulation,
    precision,
    communication,
    segmented=False,
    tracing=False,
):
    timings = defaultdict(float)
    with stage("zero_grad", backend, timings, segmented, tracing):
        optimizer.zero_grad(set_to_none=True)
    value = 0.0
    for micro in range(accumulation):
        sync = (
            function.no_sync()
            if communication == "ddp" and micro < accumulation - 1
            else nullcontext()
        )
        with sync:
            with stage("forward_and_loss", backend, timings, segmented, tracing):
                with backend.autocast(precision):
                    loss = function(*batch)
            with stage("backward_including_ddp_if_enabled", backend, timings, segmented, tracing):
                (loss / accumulation).backward()
            # XLA step markers require all profiler graph scopes to be closed.
            backend.microstep()
        with stage("host_loss_read", backend, timings, segmented, tracing):
            value += float(loss.detach().cpu())
    if communication == "per-parameter" and backend.world_size > 1:
        with stage("gradient_allreduce_and_average", backend, timings, segmented, tracing):
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.copy_(backend.reduce(parameter.grad) / backend.world_size)
    with stage("optimizer", backend, timings, segmented, tracing):
        optimizer.step()
    backend.sync()
    if not math.isfinite(value):
        raise RuntimeError("Nonfinite diagnostic loss")
    return dict(timings)


def run(args):
    torch.set_num_threads(1)
    config = load_config(args.config)
    backend = Backend(args.backend)
    settings = config.synthetic or config.imaging
    settings = settings.model_copy(
        update={
            "micro_batch_size_per_rank": args.microbatch,
            "global_batch_size": args.microbatch * args.accumulation * backend.world_size,
        }
    )
    config = config.model_copy(update={"synthetic" if config.synthetic else "imaging": settings})
    output = args.output / f"rank{backend.rank}"
    output.mkdir(parents=True, exist_ok=True)
    try:
        model = make_model(config)
        with np.load(args.initial_state, allow_pickle=False) as data:
            model.load_state_dict({k: torch.from_numpy(data[k].copy()) for k in data.files})
        initial_hash = tensor_hash(model.state_dict().items())
        model.to(backend.device).train()
        torch.set_rng_state(torch.from_numpy(np.load(args.initial_state.with_suffix(".rng.npy"))))
        batch = next(iter(original_loader(config, backend)))
        input_hash = tensor_hash((str(i), v) for i, v in enumerate(batch))
        backend.sync()
        start = time.perf_counter()
        batch = tuple(v.to(backend.device) for v in batch)
        backend.sync()
        input_transfer_seconds = time.perf_counter() - start
        function = (
            torch.compile(model, backend="tpu", fullgraph=True)
            if backend.name == "torchtpu"
            else model
        )
        if args.communication == "ddp":
            if backend.name != "torchtpu" or backend.world_size == 1:
                raise ValueError("DDP requires multi-chip TorchTPU")
            function = torch.nn.parallel.DistributedDataParallel(
                function, bucket_cap_mb=25, gradient_as_bucket_view=True
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=settings.learning_rate if config.imaging else 0.001,
            weight_decay=settings.weight_decay if config.imaging else 0.01,
            foreach=False,
            fused=False,
        )

        def step(segmented=False, tracing=False):
            return update(
                function,
                model,
                optimizer,
                batch,
                backend,
                args.accumulation,
                config.run.precision,
                args.communication,
                segmented,
                tracing,
            )

        backend.sync()
        start = time.perf_counter()
        step()
        cold_seconds = time.perf_counter() - start
        for _ in range(args.warmup):
            step()
        trials = []
        for _ in range(3):
            # Align once before each window, collect timing only after the window.
            backend.max_seconds(0)
            start = time.perf_counter()
            for _ in range(args.steps):
                step()
            local_elapsed = time.perf_counter() - start
            elapsed = backend.max_seconds(local_elapsed)
            trials.append(
                dict(
                    local_elapsed_seconds=local_elapsed,
                    slowest_rank_elapsed_seconds=elapsed,
                    samples_per_second=settings.global_batch_size * args.steps / elapsed,
                )
            )
        for _ in range(3):
            step(segmented=True)
        segmented = [step(segmented=True) for _ in range(args.segmented_steps)]
        # Separate validation-forward/CPU-read diagnostic, same cached batch.
        eval_times = []
        model.eval()
        with torch.no_grad():
            for index in range(args.segmented_steps + 3):
                backend.sync()
                start = time.perf_counter()
                with backend.autocast(config.run.precision):
                    loss = function(*batch)
                float(loss.cpu())
                backend.sync()
                if index >= 3:
                    eval_times.append(time.perf_counter() - start)
        model.train()
        result = dict(
            schema_version="tpu-phase-diagnostic-0.1",
            status="timings_complete",
            workload=config.run.workload,
            microbatch=args.microbatch,
            accumulation=args.accumulation,
            global_batch=settings.global_batch_size,
            communication=args.communication,
            initial_state_sha256=initial_hash,
            first_cpu_batch_sha256=input_hash,
            first_input_transfer_seconds=input_transfer_seconds,
            cold_first_update_seconds=cold_seconds,
            measured_updates_per_trial=args.steps,
            unsegmented_trials=trials,
            segmented_steps=segmented,
            segmented_median_ms={
                k: 1000 * statistics.median(r[k] for r in segmented) for k in segmented[0]
            },
            cached_eval_forward_and_host_read_median_ms=1000 * statistics.median(eval_times),
            runtime=backend.metadata(),
            qualifications=[
                "Diagnostic only: short fixed-update windows, no per-update timing reductions.",
                "Segmented stage fences change fusion/overlap; do not subtract these from unsegmented wall time.",
                "DDP communication is included in backward and may overlap it; only native traces can separate it.",
                "One-chip ViT uses global batch 32 to match local work with sixteen-chip global batch 512.",
                "Evaluation measures cached forward/loss/read, not full-dataset validation or metric aggregation.",
            ],
        )
        path = output / "phase-profile.json"
        path.write_text(json.dumps(result, indent=2) + "\n")
        if args.trace:
            if backend.name == "xla":
                import torch_xla.debug.profiler as xp

                xp.start_trace(str(output / "trace"))
                try:
                    for _ in range(5):
                        step(tracing=True)
                finally:
                    xp.stop_trace()
            elif backend.name == "torchtpu":
                from torch.tpu.profiler import TpuProfilerConfig

                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.TPU,
                    ],
                    experimental_config=TpuProfilerConfig(
                        run_dir=output / "trace", worker_rank=str(backend.rank)
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(str(output / "trace")),
                ) as prof:
                    for _ in range(5):
                        step(tracing=True)
                        prof.step()
        result["status"] = "completed"
        path.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        backend.close()


def worker(index, args):
    run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["xla", "torchtpu", "cpu-test"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--microbatch", type=int, default=32)
    parser.add_argument("--accumulation", type=int, default=1)
    parser.add_argument(
        "--communication", choices=["per-parameter", "ddp"], default="per-parameter"
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--segmented-steps", type=int, default=20)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--xla-spawn", action="store_true")
    args = parser.parse_args()
    if args.xla_spawn:
        import torch_xla

        torch_xla.launch(worker, args=(args,))
    else:
        run(args)
