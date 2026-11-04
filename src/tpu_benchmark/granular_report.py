"""Audit archived GPU/TPU timing and telemetry without inventing phase timings."""

import csv
import gzip
import json
import re
import statistics as st
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "results/tpu-20261028/reports"


def metric_blocks(text):
    result = {}
    for block in re.split(r"(?=^(?:Metric|Counter): )", text, flags=re.M):
        lines = block.splitlines()
        if not lines:
            continue
        name = lines[0].split(": ", 1)[-1]
        result[name] = dict(line.strip().split(": ", 1) for line in lines[1:] if ": " in line)
    return result


def phase_row(label, devices, batch, accumulation, phase, source, **extra):
    return dict(
        run=label,
        devices=devices,
        microbatch=batch,
        accumulation=accumulation,
        phase=phase.get("phase", "unknown"),
        samples_per_second=phase["throughput_samples_per_second"],
        effective_samples_per_second_per_device=phase["throughput_samples_per_second"] / devices,
        mean_update_ms=1000 * phase["elapsed_seconds"] / phase["optimizer_updates"],
        reported_step_median_ms=phase["step_time_median_ms"],
        reported_step_p95_ms=phase["step_time_p95_ms"],
        data_wait_percent=100 * phase["data_wait_seconds"] / phase["elapsed_seconds"],
        updates=phase["optimizer_updates"],
        validation_loss=phase["validation_loss"],
        source=source,
        **extra,
    )


def main():
    rows, ranks, xla = [], [], {}
    names = [
        "tx-1gpu-6-full-gpu_only",
        "imaging-8gpu-20-full-gpu_only",
        "imaging-16gpu-21-full-gpu_only",
        "imaging-8gpu-27-full-gpu_only",
        "imaging-16gpu-29-full-gpu_only",
    ]
    telemetry = {}
    for name in names:
        source = f"results/{name}/{name}.json"
        d = json.loads((ROOT / source).read_text())
        for i, phase in enumerate(d["trials"] + [d["fixed_job"]]):
            phase = {**phase, "phase": f"measurement_{i + 1}" if i < 3 else "fixed_job"}
            rows.append(
                phase_row(
                    "H100/" + name,
                    d["world_size"],
                    d["micro_batch_size_per_rank"],
                    d["gradient_accumulation_steps"],
                    phase,
                    source,
                )
            )
        buckets = defaultdict(list)
        for path in sorted(
            (ROOT / "results/archived-artifacts" / name).glob("*telemetry.rank*.jsonl.gz")
        ):
            with gzip.open(path, "rt") as f:
                for line in f:
                    d = json.loads(line)
                    for gpu in d["gpus"]:
                        buckets[(d["rank"], d["phase"], gpu["uuid"])].append(gpu)
        stats = []
        for (rank, phase, uuid), samples in sorted(buckets.items()):
            r = dict(rank=rank, phase=phase, gpu_uuid=uuid, samples=len(samples))
            for field in (
                "utilization_gpu_percent",
                "power.draw",
                "clocks.sm",
                "clocks.mem",
                "temperature.gpu",
                "memory_used_mib",
            ):
                values = [g[field] for g in samples if isinstance(g.get(field), (int, float))]
                if values:
                    r[field] = dict(mean=st.mean(values), min=min(values), max=max(values))
            stats.append(r)
        telemetry[name] = stats
    for job in (84, 85, 87):
        records = defaultdict(list)
        first_steps = {}
        for archive in sorted((ROOT / "results/tpu-20261028/collected").glob(f"job-{job}*.tar.gz")):
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    if f"results/job-{job}/" not in member.name or not member.isfile():
                        continue
                    if re.search(r"historical-(xla|torchtpu)-rank\d+\.json$", member.name):
                        d = json.load(tar.extractfile(member))
                        variant = member.name.split("/")[-2]
                        records[variant].append((d, f"{archive.relative_to(ROOT)}::{member.name}"))
                    elif member.name.endswith(".log"):
                        for line in (
                            tar.extractfile(member).read().decode(errors="replace").splitlines()
                        ):
                            if not line.startswith("{"):
                                continue
                            try:
                                d = json.loads(line)
                            except ValueError:
                                continue
                            if d.get("event") == "progress" and d.get("optimizer_updates") == 1:
                                first_steps[(Path(member.name).stem, d["phase"])] = d[
                                    "elapsed_seconds"
                                ]
        for variant, records_for_variant in sorted(records.items()):
            representative, source = next(
                (d, s) for d, s in records_for_variant if d["runtime"]["rank"] == 0
            )
            settings = representative["config"]["synthetic"] or representative["config"]["imaging"]
            n = representative["runtime"]["world_size"]
            assert sorted(d["runtime"]["rank"] for d, _ in records_for_variant) == list(range(n))
            label = f"TPU/job-{job}/{variant}"
            for phase in representative["phases"]:
                row = phase_row(
                    label,
                    n,
                    settings["micro_batch_size_per_rank"],
                    settings["global_batch_size"] // (n * settings["micro_batch_size_per_rank"]),
                    phase,
                    source,
                )
                row["first_update_seconds"] = next(
                    (
                        v
                        for (log, name), v in first_steps.items()
                        if name == phase["phase"] and log.endswith("-" + variant)
                    ),
                    None,
                )
                rows.append(row)
                per_rank = [
                    next(p for p in d["phases"] if p["phase"] == phase["phase"])
                    for d, _ in records_for_variant
                ]
                ranks.append(
                    dict(
                        run=label,
                        phase=phase["phase"],
                        ranks=n,
                        throughput_min=min(p["throughput_samples_per_second"] for p in per_rank),
                        throughput_max=max(p["throughput_samples_per_second"] for p in per_rank),
                        note="These rates use global samples and MAX-reduced elapsed time; not independent per-rank speed.",
                    )
                )
            if variant == "xla":
                xla[label] = metric_blocks(representative["runtime"]["xla_metrics"])
    out = dict(
        qualifications=[
            "GPU utilization is sampled kernel activity, not FLOP utilization or tensor-core utilization.",
            "Per-device throughput is total/world_size; distributed step timers already include gradient communication and MAX reduction.",
            "Cached-input data_wait does not measure a real storage pipeline or recurring full H2D copies.",
            "Mean update time includes loop bookkeeping and timing reductions; the reported step median excludes those later reductions. Their difference is not a measured overhead decomposition.",
            "XLA metric accumulators are runtime counters across all phases and validation. They can overlap/nest and must not be added as disjoint wall-time percentages.",
            "First update includes compilation where triggered, dispatch and execution; it is not pure compiler latency.",
            "Existing records have no isolated forward/backward/optimizer/gradient-allreduce duration or TPU MXU utilization.",
        ],
        phases=rows,
        rank_audit=ranks,
        xla_rank0_cumulative_metrics=xla,
        h100_raw_telemetry_by_rank_and_phase=telemetry,
    )
    (DEST / "granular-comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (DEST / "granular-phases.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    for label in dict.fromkeys(r["run"] for r in rows):
        trials = [r for r in rows if r["run"] == label and r["phase"].startswith("measurement_")]
        print(
            label,
            json.dumps(
                {
                    key: st.median(r[key] for r in trials)
                    for key in [
                        "samples_per_second",
                        "effective_samples_per_second_per_device",
                        "mean_update_ms",
                        "reported_step_median_ms",
                        "reported_step_p95_ms",
                        "data_wait_percent",
                    ]
                }
            ),
        )
    for label, stats in telemetry.items():
        measured = [r for r in stats if "measurement" in r["phase"]]
        print("GPU_PHASES", label, sorted({r["phase"] for r in stats}))
        if measured:
            print(
                "GPU_MEASUREMENT",
                label,
                {
                    key: {
                        "rank_phase_mean_min": min(r[key]["mean"] for r in measured),
                        "rank_phase_mean_max": max(r[key]["mean"] for r in measured),
                        "mean": st.mean(r[key]["mean"] for r in measured),
                    }
                    for key in ["utilization_gpu_percent", "power.draw", "clocks.sm"]
                },
            )
    for label, blocks in xla.items():
        print(
            "XLA_METRICS",
            label,
            {
                k: {
                    a: v
                    for a, v in blocks.get(k, {}).items()
                    if a in ("TotalSamples", "Accumulator", "Value")
                }
                for k in [
                    "CompileTime",
                    "ExecuteTime",
                    "TransferFromDeviceTime",
                    "TransferToDeviceTime",
                    "LazyTracing",
                    "UncachedCompile",
                    "xla::all_reduce",
                ]
            },
        )


if __name__ == "__main__":
    main()
