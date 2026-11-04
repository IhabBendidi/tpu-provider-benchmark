"""Compare completed TPU phases against existing AWS records, never a fresh GPU run."""

import json
import statistics

SOURCES = {
    (
        "synthetic_transcriptomics_mlp",
        1,
    ): "results/tx-1gpu-6-full-gpu_only/tx-1gpu-6-full-gpu_only.json",
    (
        "synthetic_transcriptomics_mlp",
        8,
    ): "results/tx-8gpu-7-full-gpu_only/tx-8gpu-7-full-gpu_only.json",
    (
        "synthetic_imaging_vit",
        8,
    ): "results/imaging-8gpu-20-full-gpu_only/imaging-8gpu-20-full-gpu_only.json",
    (
        "synthetic_imaging_vit",
        16,
    ): "results/imaging-16gpu-21-full-gpu_only/imaging-16gpu-21-full-gpu_only.json",
}


def compare_historical(records, repository):
    summaries = {}
    for workload, variant in sorted({key[:2] for key in records}):
        key = f"{workload}/{variant}"
        try:
            first = records[(workload, variant, 0)]
            world = first["runtime"]["world_size"]
            all_ranks = [records[(workload, variant, rank)] for rank in range(world)]
            config = first["config"]
            settings = config["synthetic"] or config["imaging"]
            schedule = ["warmup", "measurement_1", "measurement_2", "measurement_3", "fixed_job"]
            for record in all_ranks:
                if record["status"] != "completed" or record["measurement_kind"] != "full":
                    raise ValueError(
                        "Full completed runs required; smoke rates cannot rank providers"
                    )
                if (
                    record["config"] != config
                    or record["adapter_sources"] != first["adapter_sources"]
                ):
                    raise ValueError("Different configs or code between ranks")
                if record["runtime"]["world_size"] != world:
                    raise ValueError("Different world sizes")
                if [phase["phase"] for phase in record["phases"]] != schedule:
                    raise ValueError("Unexpected phase schedule")
                if len({p["initial_state_sha256"] for p in record["phases"]}) != 1:
                    raise ValueError("Initial weights changed across phases")
                if len({p["first_cpu_batch_sha256"] for p in record["phases"]}) != 1:
                    raise ValueError("Cached input changed across phases")
                for phase, reference in zip(record["phases"], first["phases"], strict=True):
                    if phase["initial_state_sha256"] != reference["initial_state_sha256"]:
                        raise ValueError("Initial weights differ across ranks")
                    if phase["optimizer_updates"] != reference["optimizer_updates"]:
                        raise ValueError("Different update counts between ranks")
            source = SOURCES[(workload, world)]
            aws = json.loads((repository / source).read_text())
            for actual, expected, name in (
                (config["run"]["seed"], aws["seed"], "seed"),
                (config["run"]["mode"], aws["mode"], "mode"),
                (config["run"]["precision"], aws["effective_precision"], "precision"),
                (settings["global_batch_size"], aws["global_batch_size"], "global batch"),
                (
                    settings["micro_batch_size_per_rank"],
                    aws["micro_batch_size_per_rank"],
                    "microbatch",
                ),
                (
                    settings["global_batch_size"]
                    // (settings["micro_batch_size_per_rank"] * world),
                    aws["gradient_accumulation_steps"],
                    "gradient accumulation",
                ),
                (config["timing"]["warmup_seconds"], 600, "warmup"),
                (config["timing"]["measurement_seconds"], 1200, "measurement duration"),
                (config["timing"]["measurement_repetitions"], len(aws["trials"]), "trials"),
                (
                    first["phases"][-1]["optimizer_updates"],
                    aws["fixed_job"]["optimizer_updates"],
                    "fixed updates",
                ),
            ):
                if actual != expected:
                    raise ValueError(f"Unmatched {name}: {actual} vs {expected}")
            for record in all_ranks:
                for index, phase in enumerate(record["phases"][:-1]):
                    target = 600 if index == 0 else 1200
                    if phase["elapsed_seconds"] < target:
                        raise ValueError("Incomplete timed phase")
                for field, value in aws["generator_parameters"].items():
                    if field in settings and settings[field] != value:
                        raise ValueError(f"Unmatched generator setting {field}")
            aws_rate = statistics.median(t["throughput_samples_per_second"] for t in aws["trials"])
            tpu_rate = min(
                statistics.median(t["throughput_samples_per_second"] for t in r["phases"][1:-1])
                for r in all_ranks
            )
            summaries[key] = {
                "status": "completed_with_aligned_historical_protocol",
                "aws_source": source,
                "device_count_each": world,
                "global_batch": aws["global_batch_size"],
                "microbatch": aws["micro_batch_size_per_rank"],
                "aws_h100_samples_per_second": aws_rate,
                "tpu_samples_per_second": tpu_rate,
                "tpu_over_h100_throughput_ratio": tpu_rate / aws_rate,
                "fixed_updates": aws["fixed_job"]["optimizer_updates"],
                "fixed_validation_loss": {
                    "aws_h100": aws["fixed_job"]["validation_loss"],
                    "tpu": first["phases"][-1]["validation_loss"],
                },
                "qualifications": first["qualifications"]
                + [
                    "Equal device counts do not mean equal chip capability, power, memory or cost.",
                    "Historical AWS input/weight identity and numerical quality equivalence remain unverified.",
                ],
            }
        except (KeyError, ValueError, FileNotFoundError) as error:
            summaries[key] = {"status": "invalid_or_incomplete", "reason": str(error)}
    return summaries
