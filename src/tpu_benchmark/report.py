"""Build cautious, all-rank comparisons directly from collected evidence archives."""

import csv
import io
import json
import tarfile
from pathlib import Path

from compare import compare
from historical_report import compare_historical
from optimization_report import summarize
from rpe1_report import summarize as summarize_rpe1


def report_job(job_id, archives, output):
    records = {}
    variants = {}
    historical = {}
    capacity = {}
    capacity_exits = {}
    rpe1 = {}
    phase_profiles = {}
    single_trials = {}
    prefix = f"results/job-{job_id}/"
    for archive_path in archives:
        with tarfile.open(archive_path) as archive:
            for member in archive:
                name = Path(member.name).name
                if (
                    member.isfile()
                    and member.name.startswith(prefix + "phase-profiles/")
                    and name == "phase-profile.json"
                ):
                    if member.size > 5_000_000:
                        raise ValueError("Unexpectedly large phase-profile record")
                    with archive.extractfile(member) as file:
                        phase_profiles[member.name.removeprefix(prefix)] = json.load(file)
                    continue
                if member.isfile() and member.name.startswith(prefix + "rpe1/"):
                    parts = Path(member.name).parts
                    if len(parts) == 6 and parts[-2].startswith("seed-"):
                        backend, seed = parts[-3], int(parts[-2].removeprefix("seed-"))
                        record = rpe1.setdefault((backend, seed), {})
                        if name in ("_SUCCESS", "configuration.json", "epochs.csv"):
                            if member.size > 5_000_000:
                                raise ValueError("Unexpectedly large RPE1 record")
                            with archive.extractfile(member) as file:
                                record[name] = (
                                    list(csv.DictReader(io.TextIOWrapper(file)))
                                    if name.endswith(".csv")
                                    else json.load(file)
                                )
                    continue
                if (
                    member.isfile()
                    and member.name.startswith(prefix)
                    and name.startswith("capacity-")
                ):
                    if name.endswith(".json"):
                        if member.size > 5_000_000:
                            raise ValueError("Unexpectedly large capacity record")
                        with archive.extractfile(member) as file:
                            capacity[name.removesuffix(".json")] = json.load(file)
                    elif name.endswith(".exit-code"):
                        with archive.extractfile(member) as file:
                            capacity_exits[name.removesuffix(".exit-code")] = int(file.read())
                    continue
                if not (
                    member.isfile()
                    and member.name.startswith(prefix)
                    and name.startswith(("xla-rank", "torchtpu-rank", "historical-"))
                    and name.endswith(".json")
                ):
                    continue
                if member.size > 5_000_000:
                    raise ValueError("Unexpectedly large result record")
                with archive.extractfile(member) as file:
                    record = json.load(file)
                key = (
                    record["config"]["run"]["workload"],
                    record["runtime"]["backend"],
                    record["runtime"]["rank"],
                )
                variant = Path(member.name).parent.name
                if name.startswith("historical-") and member.name.startswith(
                    prefix + "single-trial/"
                ):
                    single_trials[(variant, key[2])] = record
                    continue
                if name.startswith("historical-") and member.name.startswith(prefix + "preflight/"):
                    continue
                if name.startswith("historical-"):
                    historical[(key[0], variant, key[2])] = record
                    continue
                variants[(key[0], variant, key[2])] = record
                if variant not in ("xla", "torchtpu"):
                    continue
                if key in records:
                    raise ValueError(f"Duplicate result record {key}")
                records[key] = record
    workloads = {}
    for workload in ("synthetic_transcriptomics_mlp", "synthetic_imaging_vit"):
        relevant = [value for key, value in records.items() if key[0] == workload]
        if not relevant:
            workloads[workload] = {
                "status": "no_completed_records",
                "reason": "See archived backend logs and exit codes",
            }
            continue
        sizes = {record["runtime"]["world_size"] for record in relevant}
        if len(sizes) != 1:
            workloads[workload] = {"status": "invalid", "reason": "world sizes differ"}
            continue
        world_size = sizes.pop()
        pairs = []
        try:
            for rank in range(world_size):
                pairs.append(
                    compare(records[(workload, "xla", rank)], records[(workload, "torchtpu", rank)])
                )
            weights = {record["initial_state_sha256"] for record in relevant}
            if len(weights) != 1:
                raise ValueError("Initial model states differ between ranks")
        except (KeyError, ValueError) as error:
            workloads[workload] = {"status": "invalid_or_incomplete", "reason": str(error)}
        else:
            workloads[workload] = {
                "status": "completed_with_matched_inputs",
                "rank_comparisons": pairs,
            }
    result = {
        "job_id": job_id,
        "archives": [str(path) for path in archives],
        "workloads": workloads,
        "qualification": "Experimental stack comparison. Check measurement_kind and validation scope. Matching inputs does not establish convergence or optimal performance.",
    }
    if any(key[1].startswith("torchtpu-") for key in variants):
        result["optimization_variants"] = summarize(variants)
        if not any(key[1] == "xla" for key in records):
            result.pop("workloads")
    if historical:
        result["historical_comparisons"] = compare_historical(
            historical, Path(__file__).resolve().parents[2]
        )
        if not records:
            result.pop("workloads", None)
    if capacity or capacity_exits:
        for name, code in capacity_exits.items():
            if name not in capacity:
                capacity[name] = {
                    "status": "timeout_or_unrecorded_failure",
                    "exit_code": code,
                    "qualification": "No completed fit result; inspect archived log. Do not infer HBM exhaustion from this status.",
                }
            else:
                capacity[name]["exit_code"] = code
        result["capacity_candidates"] = capacity
        result.pop("workloads", None)
    if single_trials:
        from single_trial_report import summarize as summarize_single_trial

        result["single_trial_comparisons"] = summarize_single_trial(
            single_trials, Path(__file__).resolve().parents[2]
        )
        result.pop("workloads", None)
    if phase_profiles:
        result["phase_profiles"] = phase_profiles
        result.pop("workloads", None)
        result["qualification"] = (
            "Diagnostic stage fences change fusion and overlap. These are not aligned H100 throughput comparisons; native traces remain in archives."
        )
    if rpe1:
        result["rpe1_convergence"] = summarize_rpe1(rpe1, Path(__file__).resolve().parents[2])
        result.pop("workloads", None)
    output.parent.mkdir(exist_ok=True)
    temporary = output.with_suffix(".partial")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return result
