"""Distributed synthetic MLP benchmark runner, launched with torchrun."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from gpu_pilot.config import BenchmarkConfig
from gpu_pilot.contracts import (
    BenchmarkRunRecord,
    GpuTelemetry,
    RuntimeEnvironment,
    TrialMeasurement,
)
from gpu_pilot.results import write_record
from gpu_pilot.synthetic import (
    SyntheticExpressionDataset,
    SyntheticExpressionSpec,
    generator_parameters,
)
from gpu_pilot.telemetry import NvidiaSmiSampler

Precision = Literal["fp32", "bf16"]


def _log(context: DistributedContext, event: str, **values: object) -> None:
    if context.is_primary:
        sys.stdout.write(
            json.dumps({"timestamp_utc": datetime.now(UTC).isoformat(), "event": event, **values})
            + "\n"
        )
        sys.stdout.flush()


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def resolve_precision(config: BenchmarkConfig, context: DistributedContext) -> Precision:
    """Validate and return the effective training precision."""

    if config.run.precision == "fp32":
        return "fp32"
    if context.device.type != "cuda" or not torch.cuda.is_bf16_supported():
        raise RuntimeError("precision = 'bf16' requires a CUDA GPU with BF16 support")
    return "bf16"


class MaskedExpressionMlp(nn.Module):
    def __init__(self, features: int, hidden: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, hidden), nn.GELU(), nn.Linear(hidden, features)
        )

    def forward(self, values: Tensor) -> Tensor:
        return cast(Tensor, self.layers(values))

    def embed(self, values: Tensor) -> Tensor:
        return cast(Tensor, self.layers[1](self.layers[0](values)))


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return DistributedContext(
        rank=rank, world_size=world_size, local_rank=local_rank, device=device
    )


def _synchronize_max(value: float, context: DistributedContext) -> float:
    if context.world_size == 1:
        return value
    value_tensor = torch.tensor(value, device=context.device)
    dist.all_reduce(value_tensor, op=dist.ReduceOp.MAX)
    return value_tensor.item()


def collect_rank_telemetry(
    sampler: NvidiaSmiSampler, context: DistributedContext
) -> tuple[GpuTelemetry, list[GpuTelemetry], list[RuntimeEnvironment]]:
    """Stop local telemetry and gather compact summaries from every training rank."""

    local_summary = sampler.stop()
    local_runtime = sampler.runtime
    if local_runtime is None:
        raise RuntimeError("telemetry sampler did not capture runtime metadata")
    payload: dict[str, object] = {
        "telemetry": local_summary.model_dump(mode="json"),
        "runtime": local_runtime.model_dump(mode="json"),
    }
    gathered: list[object] = [payload]
    if context.world_size > 1:
        gathered = [None] * context.world_size
        dist.all_gather_object(gathered, payload)
    payloads: list[dict[str, object]] = []
    for item in gathered:
        if not isinstance(item, dict):
            raise RuntimeError("invalid gathered telemetry payload")
        payloads.append(cast(dict[str, object], item))
    summaries = sorted(
        (GpuTelemetry.model_validate(item["telemetry"]) for item in payloads),
        key=lambda item: item.rank if item.rank is not None else -1,
    )
    runtimes = sorted(
        (RuntimeEnvironment.model_validate(item["runtime"]) for item in payloads),
        key=lambda item: item.rank,
    )
    primary = next((item for item in summaries if item.rank == 0), summaries[0])
    return primary, summaries, runtimes


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * percentile))
    return ordered[index]


def _ensure_artifact_budget(
    artifact_dir: Path, state: dict[str, Tensor], maximum_bytes: int
) -> None:
    existing_bytes = sum(path.stat().st_size for path in artifact_dir.rglob("*") if path.is_file())
    state_bytes = sum(value.numel() * value.element_size() for value in state.values())
    if existing_bytes + state_bytes > maximum_bytes:
        raise RuntimeError(
            f"checkpoint would exceed artifact_max_bytes ({maximum_bytes} bytes); "
            "use a larger provider-run limit or clear old artifacts"
        )


def _make_loader(
    config: BenchmarkConfig, context: DistributedContext, validation: bool = False
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    settings = config.synthetic
    if settings is None:
        raise RuntimeError("transcriptomics runner requires [synthetic] settings")
    sample_count = (
        settings.validation_sample_count
        if validation
        else settings.dataset_size - settings.validation_sample_count
    )
    sample_offset = settings.dataset_size - settings.validation_sample_count if validation else 0
    spec = SyntheticExpressionSpec(
        seed=config.run.seed,
        dataset_size=sample_count,
        feature_count=settings.feature_count,
        latent_size=settings.latent_size,
        masking_rate=settings.masking_rate,
        sparsity_rate=settings.sparsity_rate,
        sample_offset=sample_offset,
    )
    dataset: Any
    if config.run.dataset_path is not None:
        from gpu_pilot.scratch_expression import ScratchExpressionDataset

        dataset = ScratchExpressionDataset(config.run.dataset_path, spec)
    else:
        dataset = SyntheticExpressionDataset(spec)
    sampler: DistributedSampler[tuple[Tensor, Tensor, Tensor]] | None = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=config.run.seed,
        )
        if context.world_size > 1
        else None
    )
    return DataLoader(
        dataset,
        batch_size=settings.micro_batch_size_per_rank,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=settings.num_workers,
        drop_last=True,
        pin_memory=context.device.type == "cuda",
    )


def _make_model(config: BenchmarkConfig, context: DistributedContext) -> nn.Module:
    torch.manual_seed(config.run.seed)
    settings = config.synthetic
    if settings is None:
        raise RuntimeError("transcriptomics runner requires [synthetic] settings")
    model: nn.Module = MaskedExpressionMlp(settings.feature_count, settings.hidden_size).to(
        context.device
    )
    if context.world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[context.local_rank] if context.device.type == "cuda" else None
        )
    return model


def _step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    next_batch: Callable[[], tuple[Tensor, Tensor, Tensor]],
    device: torch.device,
    accumulation: int,
    precision: Precision,
) -> tuple[float, float]:
    data_wait = 0.0
    optimizer.zero_grad(set_to_none=True)
    loss_value = 0.0
    for micro_step in range(accumulation):
        data_start = time.perf_counter()
        inputs, targets, mask = next_batch()
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        data_wait += time.perf_counter() - data_start
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if precision == "bf16"
            else nullcontext()
        )
        gradient_sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and micro_step != accumulation - 1
            else nullcontext()
        )
        with gradient_sync, autocast:
            prediction = model(inputs)
            loss = ((prediction[mask] - targets[mask]) ** 2).mean()
            (loss / accumulation).backward()
        loss_value += float(loss.detach().cpu())
    optimizer.step()
    return data_wait, loss_value / accumulation


def _validation_loss(
    model: nn.Module, config: BenchmarkConfig, context: DistributedContext, precision: Precision
) -> float:
    loader = _make_loader(config, context, validation=True)
    total_loss = torch.zeros((), device=context.device)
    total_masked_values = torch.zeros((), device=context.device)
    model.eval()
    with torch.no_grad():
        for inputs, targets, mask in loader:
            inputs, targets, mask = (
                inputs.to(context.device),
                targets.to(context.device),
                mask.to(context.device),
            )
            autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if precision == "bf16"
                else nullcontext()
            )
            with autocast:
                prediction = model(inputs)
                squared_error = (prediction[mask] - targets[mask]) ** 2
            total_loss += squared_error.float().sum()
            total_masked_values += mask.sum()
    if context.world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_masked_values, op=dist.ReduceOp.SUM)
    model.train()
    return float((total_loss / total_masked_values).cpu())


def _run_trial(
    config: BenchmarkConfig,
    context: DistributedContext,
    seconds: float | None,
    updates: int | None,
    gpu_only: bool,
    accumulation: int,
    precision: Precision,
    retain_state: bool = False,
    phase: str = "trial",
) -> tuple[TrialMeasurement, dict[str, Tensor] | None]:
    settings = config.synthetic
    if settings is None:
        raise RuntimeError("transcriptomics runner requires [synthetic] settings")
    _log(context, "phase_start", phase=phase, seconds=seconds, target_updates=updates)
    model = _make_model(config, context)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = _make_loader(config, context)
    iterator = iter(loader)
    preloaded_batch: tuple[Tensor, Tensor, Tensor] | None = None
    if gpu_only:
        source_batch = cast(tuple[Tensor, Tensor, Tensor], next(iterator))
        preloaded_batch = cast(
            tuple[Tensor, Tensor, Tensor],
            tuple(value.to(context.device) for value in source_batch),
        )

    def next_batch() -> tuple[Tensor, Tensor, Tensor]:
        nonlocal iterator
        if preloaded_batch is not None:
            return preloaded_batch
        try:
            return cast(tuple[Tensor, Tensor, Tensor], next(iterator))
        except StopIteration:
            iterator = iter(loader)
            return cast(tuple[Tensor, Tensor, Tensor], next(iterator))

    if context.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(context.device)
        torch.cuda.synchronize(context.device)
    started = time.perf_counter()
    step_times: list[float] = []
    waits: list[float] = []
    losses: list[float] = []
    count = 0
    elapsed = 0.0
    last_log = 0.0
    # All ranks use the same reduced elapsed time to leave timed loops together.
    while (seconds is not None and elapsed < seconds) or (updates is not None and count < updates):
        step_started = time.perf_counter()
        wait, loss = _step(
            model,
            optimizer,
            next_batch,
            context.device,
            accumulation,
            precision,
        )
        if context.device.type == "cuda":
            torch.cuda.synchronize(context.device)
        step_times.append(_synchronize_max(time.perf_counter() - step_started, context))
        waits.append(_synchronize_max(wait, context))
        losses.append(loss)
        count += 1
        elapsed = _synchronize_max(time.perf_counter() - started, context)
        if count == 1 or elapsed - last_log >= 5:
            _log(
                context,
                "progress",
                phase=phase,
                optimizer_updates=count,
                elapsed_seconds=elapsed,
                throughput_samples_per_second=count * settings.global_batch_size / elapsed,
                step_time_ms=step_times[-1] * 1000,
                training_loss=loss,
                data_wait_seconds=sum(waits),
                peak_gpu_memory_bytes=(
                    torch.cuda.max_memory_allocated(context.device)
                    if context.device.type == "cuda"
                    else 0
                ),
            )
            last_log = elapsed
    peak_memory = (
        torch.cuda.max_memory_allocated(context.device) if context.device.type == "cuda" else 0
    )
    peak_memory = int(_synchronize_max(float(peak_memory), context))
    samples = count * settings.global_batch_size
    _log(context, "validation_start", phase=phase, optimizer_updates=count)
    measurement = TrialMeasurement(
        elapsed_seconds=elapsed,
        optimizer_updates=count,
        samples_processed=samples,
        throughput_samples_per_second=samples / elapsed,
        step_time_median_ms=_percentile(step_times, 0.5) * 1000,
        step_time_p95_ms=_percentile(step_times, 0.95) * 1000,
        data_wait_seconds=sum(waits),
        peak_gpu_memory_bytes=peak_memory,
        validation_loss=_validation_loss(model, config, context, precision),
    )
    _log(context, "phase_complete", phase=phase, **measurement.model_dump(mode="json"))
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    state = (
        {name: value.detach().cpu() for name, value in raw_model.state_dict().items()}
        if retain_state and context.is_primary
        else None
    )
    return measurement, state


def execute(config: BenchmarkConfig, profile_name: str) -> Path | None:
    """Execute all benchmark sections; only rank zero writes a result record."""

    if config.run.workload == "synthetic_imaging_vit":
        from gpu_pilot.imaging import execute_imaging

        return execute_imaging(config, profile_name)

    context = initialize_distributed()
    settings = config.synthetic
    if settings is None:
        raise RuntimeError("transcriptomics runner requires [synthetic] settings")
    if context.world_size != settings.expected_world_size:
        raise RuntimeError(
            f"expected world size {settings.expected_world_size}, got {context.world_size}"
        )
    accumulation = settings.gradient_accumulation_steps(context.world_size)
    precision = resolve_precision(config, context)
    _log(
        context,
        "run_start",
        world_size=context.world_size,
        precision=precision,
        gradient_accumulation_steps=accumulation,
        config=config.model_dump(mode="json"),
    )
    started_at = datetime.now(UTC)
    sampler = NvidiaSmiSampler(
        artifact_path=config.run.artifact_dir
        / f"{config.run.run_id}.telemetry.rank{context.rank}.jsonl",
        rank=context.rank,
        stdout=True,
    )
    sampler.start()
    try:
        sampler.set_phase("warmup")
        _run_trial(
            config,
            context,
            seconds=config.timing.warmup_seconds,
            updates=None,
            gpu_only=config.run.mode == "gpu_only",
            accumulation=accumulation,
            precision=precision,
            phase="warmup",
        )
        trials = []
        for index in range(config.timing.measurement_repetitions):
            sampler.set_phase(f"measurement_{index + 1}")
            trials.append(
                _run_trial(
                    config,
                    context,
                    seconds=config.timing.measurement_seconds,
                    updates=None,
                    gpu_only=config.run.mode == "gpu_only",
                    accumulation=accumulation,
                    precision=precision,
                    phase=f"measurement_{index + 1}",
                )[0]
            )
        sampler.set_phase("fixed_job")
        fixed_job, state = _run_trial(
            config,
            context,
            seconds=None,
            updates=config.timing.fixed_job_updates,
            gpu_only=config.run.mode == "gpu_only",
            accumulation=accumulation,
            precision=precision,
            retain_state=True,
            phase="fixed_job",
        )
        telemetry, telemetry_by_rank, runtime_environments = collect_rank_telemetry(
            sampler, context
        )
        if not context.is_primary:
            return None
        if state is None:
            raise RuntimeError("rank zero did not retain the fixed-job model state")
        config.run.artifact_dir.mkdir(parents=True, exist_ok=True)
        _ensure_artifact_budget(config.run.artifact_dir, state, config.run.artifact_max_bytes)
        checkpoint_started = time.perf_counter()
        torch.save(state, config.run.artifact_dir / f"{config.run.run_id}.pt")
        checkpoint_seconds = time.perf_counter() - checkpoint_started
        export_started = time.perf_counter()
        (config.run.artifact_dir / f"{config.run.run_id}.export.json").write_text(
            '{"format":"gpu-pilot-synthetic-expression-v1"}\n', encoding="utf-8"
        )
        export_seconds = time.perf_counter() - export_started
        spec = SyntheticExpressionSpec(
            seed=config.run.seed,
            dataset_size=settings.dataset_size,
            feature_count=settings.feature_count,
            latent_size=settings.latent_size,
            masking_rate=settings.masking_rate,
            sparsity_rate=settings.sparsity_rate,
        )
        diagnostics = [telemetry.diagnostic] if telemetry.diagnostic else []
        record = BenchmarkRunRecord(
            run_id=config.run.run_id,
            workload=config.run.workload,
            mode=config.run.mode,
            profile=profile_name,
            started_at_utc=started_at,
            provider=config.provider,
            billing=config.billing,
            world_size=context.world_size,
            seed=config.run.seed,
            requested_precision=config.run.precision,
            effective_precision=precision,
            global_batch_size=settings.global_batch_size,
            micro_batch_size_per_rank=settings.micro_batch_size_per_rank,
            gradient_accumulation_steps=accumulation,
            generator="synthetic_expression_v1",
            generator_parameters={
                **generator_parameters(spec),
                "storage": "scratch_direct_io" if config.run.dataset_path else "indexed_on_the_fly",
                "validation_sample_count": settings.validation_sample_count,
            },
            trials=trials,
            fixed_job=fixed_job,
            gpu_telemetry=telemetry,
            telemetry_by_rank=telemetry_by_rank,
            runtime_environments=runtime_environments,
            checkpoint_seconds=checkpoint_seconds,
            export_seconds=export_seconds,
            diagnostics=diagnostics,
        )
        path = write_record(record, config.run.output_dir)
        throughputs = sorted(trial.throughput_samples_per_second for trial in trials)
        median = throughputs[len(throughputs) // 2]
        _log(
            context,
            "run_complete",
            result_path=str(path),
            median_throughput_samples_per_second=median,
            throughput_range=[throughputs[0], throughputs[-1]],
            compute_cost_per_million_samples=(
                config.billing.configuration_hourly_cost * 1_000_000 / (3600 * median)
                if config.billing.configuration_hourly_cost
                else None
            ),
            cost_status="observed_input"
            if config.billing.configuration_hourly_cost
            else "price_missing",
            record=record.model_dump(mode="json"),
        )
        return path
    finally:
        if sampler._thread is not None and sampler._thread.is_alive():
            sampler.stop()
        if context.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()
