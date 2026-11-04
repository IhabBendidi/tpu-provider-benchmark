"""One measured trial compared with the first archived H100 trial only."""

import json


def summarize(records, root):
    aws_path = "results/imaging-16gpu-21-full-gpu_only/imaging-16gpu-21-full-gpu_only.json"
    if not (root / aws_path).exists():
        return {"status": "reference_unavailable", "reason": "Optional AWS reference data are not included in this code-only repository"}
    aws = json.loads((root / aws_path).read_text())
    output = {}
    for variant in sorted({key[0] for key in records}):
        ranks = [r for (v, _), r in records.items() if v == variant]
        try:
            assert sorted(r["runtime"]["rank"] for r in ranks) == list(range(16)), "Missing ranks"
            assert {r["status"] for r in ranks} == {"completed"}, "Incomplete records"
            assert {r["runtime"]["world_size"] for r in ranks} == {16}, "Wrong world size"
            assert len({r["optimization"] for r in ranks}) == 1
            config = ranks[0]["config"]
            imaging = config["imaging"]
            assert (
                imaging["micro_batch_size_per_rank"] == 32 and imaging["global_batch_size"] == 512
            )
            assert config["run"]["precision"] == "bf16"
            assert config["run"]["seed"] == aws["seed"]
            expected_architecture = dict(
                image_size=224,
                patch_size=16,
                embedding_size=768,
                transformer_depth=12,
                attention_heads=12,
                mlp_size=3072,
                class_count=12,
                learning_rate=0.001,
                weight_decay=0.01,
                expected_world_size=16,
            )
            assert all(imaging[k] == v for k, v in expected_architecture.items()), (
                "Architecture or optimizer settings changed"
            )
            assert len({r["initial_state_sha256"] for d in ranks for r in d["phases"]}) == 1
            trials = []
            for record in ranks:
                assert record["single_trial"]
                assert [r["phase"] for r in record["phases"]] == ["warmup", "measurement_1"]
                assert record["config"] == config
                assert record["actual_schedule"] == [
                    ["warmup", 30.0, None],
                    ["measurement_1", 1200.0, None],
                ]
                trial = record["phases"][1]
                assert trial["elapsed_seconds"] >= 1200
                trials.append(trial)
            assert len({r["optimizer_updates"] for r in trials}) == 1
            trial = trials[0]
            rate = trial["throughput_samples_per_second"]
            output[variant] = dict(
                status="completed_single_trial_all_ranks",
                tpu_samples_per_second=rate,
                h100_trial_1_samples_per_second=aws["trials"][0]["throughput_samples_per_second"],
                tpu_over_h100_trial_1=rate / aws["trials"][0]["throughput_samples_per_second"],
                trial=trial,
                aws_source=aws_path,
                qualifications=[
                    "One 1200-second trial on each side; no selection of fastest trial.",
                    "TPU uses a 30-second warmup versus historical 600 seconds; both reset model and optimizer before measurement.",
                    "Per-microbatch host loss read, per-update synchronization and three MAX timing reductions are preserved.",
                    "No fixed-update convergence comparison: unequal update counts in timed trials prevent validation-loss parity claims.",
                    "Candidate uses backend-specific optimization against the archived H100 implementation.",
                ],
            )
        except (AssertionError, KeyError) as error:
            output[variant] = dict(status="invalid_or_incomplete", reason=str(error))
    return output
