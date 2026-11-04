"""Stable, provider-neutral benchmark result contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

RUN_RECORD_SCHEMA_VERSION: Final[Literal["1.0"]] = "1.0"


class ProviderConfiguration(BaseModel):
    """Exact rented configuration, recorded for every measured run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    region: str = Field(min_length=1)
    instance_type: str = Field(min_length=1)
    gpu_model: str = Field(min_length=1)
    gpu_count: PositiveInt
    node_count: PositiveInt
    network_description: str = Field(min_length=1)
    storage_description: str = Field(min_length=1)


class BillingObservation(BaseModel):
    """Observed, time-stamped costs used in benchmark calculations."""

    model_config = ConfigDict(extra="forbid")

    currency: str = Field(min_length=3, max_length=3)
    configuration_hourly_cost: float = Field(ge=0)
    tax_included: bool
    price_source: str = Field(min_length=1)
    observed_at_utc: datetime


class TrialMeasurement(BaseModel):
    """Metrics for one timed measurement window."""

    model_config = ConfigDict(extra="forbid")

    elapsed_seconds: float = Field(gt=0)
    optimizer_updates: PositiveInt
    samples_processed: PositiveInt
    throughput_samples_per_second: float = Field(gt=0)
    step_time_median_ms: float = Field(gt=0)
    step_time_p95_ms: float = Field(gt=0)
    data_wait_seconds: float = Field(ge=0)
    peak_gpu_memory_bytes: int = Field(ge=0)
    validation_loss: float = Field(ge=0)


class GpuTelemetry(BaseModel):
    """Best-effort per-rank training telemetry summary."""

    model_config = ConfigDict(extra="forbid")

    sample_count: int = Field(ge=0)
    utilization_percent_mean: float | None = Field(default=None, ge=0, le=100)
    memory_used_mib_max: float | None = Field(default=None, ge=0)
    utilization_percent_max: float | None = Field(default=None, ge=0, le=100)
    utilization_percent_p95: float | None = Field(default=None, ge=0, le=100)
    cpu_utilization_percent_max: float | None = Field(default=None, ge=0, le=100)
    host_memory_used_mib_max: float | None = Field(default=None, ge=0)
    rank: int | None = Field(default=None, ge=0)
    hostname: str | None = None
    raw_artifact_path: str | None = None
    diagnostic: str | None = None
    query_retry_count: int = Field(default=0, ge=0)


class RuntimeEnvironment(BaseModel):
    """Versioned hardware/runtime identity captured from one training rank."""

    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=0)
    hostname: str = Field(min_length=1)
    nvidia_driver_version: str | None = None
    nvidia_smi_version: str | None = None
    cuda_driver_version: str | None = None
    pytorch_version: str = Field(min_length=1)
    pytorch_cuda_version: str | None = None
    gpu_uuids: list[str] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)


class BenchmarkRunRecord(BaseModel):
    """The interoperable record emitted for each completed benchmark run."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = RUN_RECORD_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    workload: str = Field(min_length=1)
    mode: Literal["gpu_only", "real_pipeline"]
    profile: str = Field(min_length=1)
    started_at_utc: datetime
    provider: ProviderConfiguration
    billing: BillingObservation
    world_size: PositiveInt
    seed: int
    requested_precision: Literal["fp32", "bf16"]
    effective_precision: Literal["fp32", "bf16"]
    global_batch_size: PositiveInt
    micro_batch_size_per_rank: PositiveInt
    gradient_accumulation_steps: PositiveInt
    generator: str = Field(min_length=1)
    generator_parameters: dict[str, int | float | str | bool]
    trials: list[TrialMeasurement] = Field(min_length=1)
    fixed_job: TrialMeasurement
    gpu_telemetry: GpuTelemetry
    telemetry_by_rank: list[GpuTelemetry] = Field(default_factory=list)
    runtime_environments: list[RuntimeEnvironment] = Field(default_factory=list)
    checkpoint_seconds: float = Field(ge=0)
    export_seconds: float = Field(ge=0)
    diagnostics: list[str]


class InferenceMeasurement(BaseModel):
    """One batch-size/mode measurement from a frozen-checkpoint inference run."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["gpu_only", "end_to_end"]
    batch_size: PositiveInt
    sample_count: PositiveInt
    embedding_dimension: PositiveInt
    elapsed_seconds: float = Field(gt=0)
    throughput_embeddings_per_second: float = Field(gt=0)
    batch_one_latency_median_ms: float | None = Field(default=None, gt=0)
    batch_one_latency_p95_ms: float | None = Field(default=None, gt=0)
    total_job_cost: float = Field(ge=0)
    cost_per_million_embeddings: float = Field(ge=0)
    output_path: str | None = None


class InferenceRunRecord(BaseModel):
    """Portable record for frozen-checkpoint embedding inference."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = RUN_RECORD_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    workload: str = Field(min_length=1)
    started_at_utc: datetime
    provider: ProviderConfiguration
    billing: BillingObservation
    seed: int
    precision: Literal["fp32", "bf16"]
    checkpoint_sha256: str = Field(min_length=64, max_length=64)
    checkpoint_path: str = Field(min_length=1)
    generator: str = Field(min_length=1)
    generator_parameters: dict[str, int | float | str | bool]
    measurements: list[InferenceMeasurement] = Field(min_length=1)
    diagnostics: list[str]


class TransferRunRecord(BaseModel):
    """Versioned record for one disposable storage/transfer benchmark."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["1.0"] = RUN_RECORD_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    started_at_utc: datetime
    profile: Literal["local", "provider_full"]
    direction: Literal["upload", "download"]
    layout: Literal["shards", "small_files"]
    workers: PositiveInt
    byte_count: PositiveInt
    object_count: PositiveInt
    elapsed_seconds: float = Field(gt=0)
    throughput_mb_per_second: float = Field(gt=0)
    files_per_second: float = Field(ge=0)
    list_seconds: float = Field(ge=0)
    estimated_cost_eur: float = Field(ge=0)
    remote_prefix: str = Field(min_length=1)
    cleanup_performed: bool
    diagnostics: list[str]
