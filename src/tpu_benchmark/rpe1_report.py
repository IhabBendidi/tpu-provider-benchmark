"""All-seed RPE1 convergence checks against the archived 150-epoch H100 runs."""

import csv
import json
import statistics


def summarize(records, repository):
    output = {}
    aws_root = repository / "results/rpe1-convergence/saturation-20261021T160206Z/final"
    for backend in sorted({key[0] for key in records}):
        comparisons = []
        try:
            for seed in range(1000, 1016):
                record = records[(backend, seed)]
                summary = record["_SUCCESS"]
                config = record["configuration.json"]
                if summary["phase"] != "final":
                    raise ValueError(
                        "Smoke result; full convergence comparison requires 150 epochs"
                    )
                rows = record["epochs.csv"]
                if [int(row["epoch"]) for row in rows] != list(range(151)):
                    raise ValueError(f"Incomplete epoch history for seed {seed}")
                if summary["status"] != "complete" or summary["epochs"] != 150:
                    raise ValueError(f"Incomplete seed {seed}")
                if (
                    int(rows[-1]["optimizer_updates"]) != 51300
                    or summary["optimizer_updates"] != 51300
                ):
                    raise ValueError(f"Unmatched update count for seed {seed}")
                if config["matmul_precision"] != "highest" or not config["independent_seed_no_ddp"]:
                    raise ValueError("Different precision or seed-parallelism policy")
                aws_path = aws_root / f"seed-{seed}"
                aws_config = json.loads((aws_path / "configuration.json").read_text())
                for field in (
                    "seed",
                    "batch_size",
                    "learning_rate",
                    "data_manifest_sha256",
                    "features",
                    "perturbations",
                    "encoder_sizes",
                    "decoder_sizes",
                    "parameter_count",
                    "precision",
                    "optimizer",
                    "betas",
                    "epsilon",
                    "weight_decay",
                    "gradient_accumulation",
                    "training_rows",
                ):
                    if config[field] != aws_config[field]:
                        raise ValueError(f"Unmatched {field} for seed {seed}")
                with (aws_path / "epochs.csv").open() as file:
                    aws_rows = list(csv.DictReader(file))
                if [int(row["epoch"]) for row in aws_rows] != list(range(151)):
                    raise ValueError("Archived AWS epoch history is incomplete")
                a, b = aws_rows[-1], rows[-1]
                targets = {}
                for threshold in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
                    targets[str(threshold)] = {}
                    for label, series in (("aws_h100", aws_rows), (backend, rows)):
                        hit = next(
                            (
                                r
                                for r in series
                                if float(r["validation_pearson_delta"]) >= threshold
                            ),
                            None,
                        )
                        targets[str(threshold)][label] = (
                            None
                            if hit is None
                            else {
                                "epoch": int(hit["epoch"]),
                                "updates": int(hit["optimizer_updates"]),
                                "training_seconds": float(hit["cumulative_train_seconds"]),
                                "wall_seconds_including_evaluation": float(hit["wall_seconds"]),
                            }
                        )
                comparisons.append(
                    {
                        "seed": seed,
                        "aws_source": str(aws_path.relative_to(repository)),
                        "initial_state_sha256": config["initial_state_sha256"],
                        "batch_stream_sha256_by_epoch": [
                            row["batch_stream_sha256"] for row in rows[1:]
                        ],
                        "final_validation_mse": {
                            "aws_h100": float(a["validation_mse"]),
                            backend: float(b["validation_mse"]),
                        },
                        "final_validation_pearson_delta": {
                            "aws_h100": float(a["validation_pearson_delta"]),
                            backend: float(b["validation_pearson_delta"]),
                        },
                        "final_test_pearson_delta": {
                            "aws_h100": float(a["test_pearson_delta"]),
                            backend: float(b["test_pearson_delta"]),
                        },
                        "training_seconds": {
                            "aws_h100": float(a["cumulative_train_seconds"]),
                            backend: float(b["cumulative_train_seconds"]),
                        },
                        "wall_seconds_including_evaluation": {
                            "aws_h100": float(a["wall_seconds"]),
                            backend: float(b["wall_seconds"]),
                        },
                        "epoch_zero_validation_mse_difference": float(rows[0]["validation_mse"])
                        - float(aws_rows[0]["validation_mse"]),
                        "validation_target_crossings": targets,
                    }
                )
            means = {
                field: {
                    label: statistics.mean(item[field][label] for item in comparisons)
                    for label in ("aws_h100", backend)
                }
                for field in (
                    "final_validation_mse",
                    "final_validation_pearson_delta",
                    "training_seconds",
                    "wall_seconds_including_evaluation",
                )
            }
            output[backend] = {
                "status": "completed_with_aligned_rpe1_protocol",
                "seeds": 16,
                "epochs": 150,
                "batch_per_device": 512,
                "means": means,
                "seed_comparisons": comparisons,
                "qualification": "Independent single-device seeds, FP32 tensors with highest TPU matmul precision; convergence equivalence must be assessed from curves, not assumed. Compilation is included in startup/training/evaluation times where incurred; not separately isolated.",
            }
        except (KeyError, ValueError, FileNotFoundError) as error:
            output[backend] = {
                "status": "smoke_or_incomplete",
                "reason": str(error),
                "completed_seed_count": sum(
                    "_SUCCESS" in v for k, v in records.items() if k[0] == backend
                ),
            }
    return output
