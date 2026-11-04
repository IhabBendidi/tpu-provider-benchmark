"""Verify every rank before summarizing TorchTPU execution variants."""

import statistics


def summarize(records):
    result = {}
    for workload in sorted({key[0] for key in records}):
        variants = sorted({key[1] for key in records if key[0] == workload})
        summaries = {}
        reference = records.get((workload, "torchtpu", 0))
        for variant in variants:
            try:
                if reference is None:
                    raise ValueError("Missing reference rank 0")
                rates, losses, validations, warmups = [], [], [], []
                for rank in range(reference["runtime"]["world_size"]):
                    a = records[(workload, "torchtpu", rank)]
                    b = records[(workload, variant, rank)]
                    for field in (
                        "status",
                        "config_sha256",
                        "initial_state_sha256",
                        "first_cpu_batch_sha256",
                        "adapter_sources",
                        "total_optimizer_updates",
                        "warmup_updates",
                        "gradient_accumulation_steps",
                    ):
                        if a[field] != b[field]:
                            raise ValueError(f"Unmatched {field}, rank {rank}")
                    if a["status"] != "completed":
                        raise ValueError("Incomplete record")
                    for field in ("world_size", "rank", "packages"):
                        if a["runtime"][field] != b["runtime"][field]:
                            raise ValueError(f"Unmatched runtime {field}")
                    for left, right in zip(a["trials"], b["trials"], strict=True):
                        for field in ("optimizer_updates", "samples_processed"):
                            if left[field] != right[field]:
                                raise ValueError(f"Unmatched trial {field}")
                        losses.extend(
                            abs(x - y) for x, y in zip(left["losses"], right["losses"], strict=True)
                        )
                    for field in ("scope", "denominator", "batches_per_rank"):
                        if a["validation"][field] != b["validation"][field]:
                            raise ValueError(f"Unmatched validation {field}")
                    rates.append(
                        statistics.median(t["throughput_samples_per_second"] for t in b["trials"])
                    )
                    validations.append(b["validation"]["loss"])
                    warmups.append(b["warmup_seconds"])
                baseline = statistics.median(
                    t["throughput_samples_per_second"] for t in reference["trials"]
                )
                summaries[variant] = {
                    "status": "completed_with_matched_inputs",
                    "world_size": len(rates),
                    "samples_per_second": min(rates),
                    "speedup_over_reference": min(rates) / baseline,
                    "max_absolute_training_loss_difference": max(losses),
                    "validation_loss": max(validations),
                    "warmup_seconds": max(warmups),
                    "communication": b.get("communication", "per-parameter"),
                    "defer_and_fuse": b["runtime"]["launch_environment"].get(
                        "TORCH_TPU_DEFER_AND_FUSE", "0"
                    ),
                    "trial_seconds": [t["elapsed_seconds"] for t in b["trials"]],
                }
            except (KeyError, ValueError) as error:
                summaries[variant] = {"status": "invalid_or_incomplete", "reason": str(error)}
        result[workload] = summaries
    return result
