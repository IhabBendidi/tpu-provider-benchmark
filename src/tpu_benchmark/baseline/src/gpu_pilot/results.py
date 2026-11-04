"""Atomic JSON result writing and flat CSV summarization."""

from __future__ import annotations

import csv
from pathlib import Path

from gpu_pilot.contracts import BenchmarkRunRecord, InferenceRunRecord


def write_record(record: BenchmarkRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{record.run_id}.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def write_inference_record(record: InferenceRunRecord, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{record.run_id}.inference.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def summarize_records(input_dir: Path, output_path: Path) -> int:
    records = [
        BenchmarkRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(input_dir.glob("*.json"))
        if not path.name.endswith(".inference.json")
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "workload",
        "mode",
        "profile",
        "provider",
        "region",
        "gpu_count",
        "node_count",
        "world_size",
        "effective_precision",
        "global_batch_size",
        "micro_batch_size_per_rank",
        "gradient_accumulation_steps",
        "trial_median_throughput_samples_per_second",
        "trial_range_throughput_samples_per_second",
        "trial_median_step_time_ms",
        "trial_p95_step_time_ms",
        "fixed_job_throughput_samples_per_second",
        "nvidia_driver_versions",
        "max_nvidia_memory_used_mib",
        "max_gpu_utilization_percent",
        "max_cpu_utilization_percent",
        "max_host_memory_used_mib",
        "configuration_hourly_cost",
        "compute_cost_per_million_samples",
        "currency",
        "diagnostics",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            throughputs = [trial.throughput_samples_per_second for trial in record.trials]
            median_throughput = sorted(throughputs)[len(throughputs) // 2]
            writer.writerow(
                {
                    "run_id": record.run_id,
                    "workload": record.workload,
                    "mode": record.mode,
                    "profile": record.profile,
                    "provider": record.provider.name,
                    "region": record.provider.region,
                    "gpu_count": record.provider.gpu_count,
                    "node_count": record.provider.node_count,
                    "world_size": record.world_size,
                    "effective_precision": record.effective_precision,
                    "global_batch_size": record.global_batch_size,
                    "micro_batch_size_per_rank": record.micro_batch_size_per_rank,
                    "gradient_accumulation_steps": record.gradient_accumulation_steps,
                    "trial_median_throughput_samples_per_second": median_throughput,
                    "trial_range_throughput_samples_per_second": f"{min(throughputs):.3f}-{max(throughputs):.3f}",
                    "trial_median_step_time_ms": sorted(
                        t.step_time_median_ms for t in record.trials
                    )[len(record.trials) // 2],
                    "trial_p95_step_time_ms": max(t.step_time_p95_ms for t in record.trials),
                    "fixed_job_throughput_samples_per_second": record.fixed_job.throughput_samples_per_second,
                    "nvidia_driver_versions": " | ".join(
                        sorted(
                            {
                                environment.nvidia_driver_version
                                for environment in record.runtime_environments
                                if environment.nvidia_driver_version is not None
                            }
                        )
                    ),
                    "max_nvidia_memory_used_mib": max(
                        (item.memory_used_mib_max or 0 for item in record.telemetry_by_rank),
                        default=record.gpu_telemetry.memory_used_mib_max or 0,
                    ),
                    "max_gpu_utilization_percent": max(
                        (item.utilization_percent_max or 0 for item in record.telemetry_by_rank),
                        default=record.gpu_telemetry.utilization_percent_max or 0,
                    ),
                    "max_cpu_utilization_percent": max(
                        (
                            item.cpu_utilization_percent_max or 0
                            for item in record.telemetry_by_rank
                        ),
                        default=record.gpu_telemetry.cpu_utilization_percent_max or 0,
                    ),
                    "max_host_memory_used_mib": max(
                        (item.host_memory_used_mib_max or 0 for item in record.telemetry_by_rank),
                        default=record.gpu_telemetry.host_memory_used_mib_max or 0,
                    ),
                    "configuration_hourly_cost": record.billing.configuration_hourly_cost,
                    "compute_cost_per_million_samples": record.billing.configuration_hourly_cost
                    * 1_000_000
                    / (3600 * median_throughput),
                    "currency": record.billing.currency,
                    "diagnostics": " | ".join(record.diagnostics),
                }
            )
    return len(records)


def summarize_inference_records(input_dir: Path, output_path: Path) -> int:
    records = [
        InferenceRunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(input_dir.glob("*.inference.json"))
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "workload",
        "provider",
        "region",
        "precision",
        "batch_size",
        "mode",
        "sample_count",
        "embedding_dimension",
        "throughput_embeddings_per_second",
        "batch_one_latency_median_ms",
        "batch_one_latency_p95_ms",
        "total_job_cost",
        "cost_per_million_embeddings",
        "currency",
        "output_path",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            for measurement in record.measurements:
                writer.writerow(
                    {
                        "run_id": record.run_id,
                        "workload": record.workload,
                        "provider": record.provider.name,
                        "region": record.provider.region,
                        "precision": record.precision,
                        "batch_size": measurement.batch_size,
                        "mode": measurement.mode,
                        "sample_count": measurement.sample_count,
                        "embedding_dimension": measurement.embedding_dimension,
                        "throughput_embeddings_per_second": measurement.throughput_embeddings_per_second,
                        "batch_one_latency_median_ms": measurement.batch_one_latency_median_ms,
                        "batch_one_latency_p95_ms": measurement.batch_one_latency_p95_ms,
                        "total_job_cost": measurement.total_job_cost,
                        "cost_per_million_embeddings": measurement.cost_per_million_embeddings,
                        "currency": record.billing.currency,
                        "output_path": measurement.output_path,
                    }
                )
    return len(records)
