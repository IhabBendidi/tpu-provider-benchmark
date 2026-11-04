"""Compare only completed, matched PyTorch/XLA and TorchTPU records."""

import argparse
import json
import math
import statistics
from pathlib import Path


def compare(xla, tpu):
    for field in ("baseline_commit", "adapter_sources", "measurement_kind", "communication"):
        if xla.get(field) != tpu.get(field):
            raise ValueError(f"Unmatched {field}")
    for record, backend in ((xla, "xla"), (tpu, "torchtpu")):
        if record["status"] != "completed" or record["runtime"]["backend"] != backend:
            raise ValueError("Requires completed XLA and TorchTPU runs")
    for field in (
        "config_sha256",
        "initial_state_sha256",
        "first_cpu_batch_sha256",
        "gradient_accumulation_steps",
        "total_optimizer_updates",
        "warmup_updates",
    ):
        if xla[field] != tpu[field]:
            raise ValueError(f"Unmatched {field}")
    for field in ("world_size", "rank"):
        if xla["runtime"][field] != tpu["runtime"][field]:
            raise ValueError(f"Unmatched {field}")
    if len(xla["trials"]) != len(tpu["trials"]):
        raise ValueError("Unmatched trial count")
    for left, right in zip(xla["trials"], tpu["trials"], strict=True):
        for field in ("optimizer_updates", "samples_processed"):
            if left[field] != right[field]:
                raise ValueError(f"Unmatched trial {field}")
    for field in ("scope", "denominator", "batches_per_rank"):
        if xla["validation"][field] != tpu["validation"][field]:
            raise ValueError(f"Unmatched validation {field}")
    a = [v for trial in xla["trials"] for v in trial["losses"]]
    b = [v for trial in tpu["trials"] for v in trial["losses"]]
    if len(a) != len(b) or not all(math.isfinite(value) for value in a + b):
        raise ValueError("Invalid or unmatched loss series")
    throughputs = {
        label: statistics.median(t["throughput_samples_per_second"] for t in r["trials"])
        for label, r in (("xla", xla), ("torchtpu", tpu))
    }
    return {
        "matched_inputs_and_initialization": True,
        "world_size": xla["runtime"]["world_size"],
        "workload": xla["config"]["run"]["workload"],
        "precision": xla["config"]["run"]["precision"],
        "measured_updates_per_backend": len(a),
        "throughput_samples_per_second": throughputs,
        "torchtpu_over_xla_throughput_ratio": throughputs["torchtpu"] / throughputs["xla"],
        "max_absolute_training_loss_difference": max(
            abs(left - right) for left, right in zip(a, b, strict=True)
        ),
        "validation_loss": {
            "xla": xla["validation"]["loss"],
            "torchtpu": tpu["validation"]["loss"],
        },
        "scope": xla["validation"]["scope"],
        "runtime_versions": {
            "xla": xla["runtime"]["packages"],
            "torchtpu": tpu["runtime"]["packages"],
        },
        "measurement_kind": xla.get("measurement_kind", "smoke"),
        "qualification": "Stack comparison with different PyTorch/libtpu versions; not a provider ranking or convergence study.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xla", type=Path, required=True)
    parser.add_argument("--torchtpu", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.xla.read_text()), json.loads(args.torchtpu.read_text()))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))
