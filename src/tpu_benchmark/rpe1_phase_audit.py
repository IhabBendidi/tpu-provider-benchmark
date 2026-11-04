"""Read completed RPE1 epochs 2-20; no device initialization or training changes."""

import csv
import gzip
import json
import statistics as st
import sys
from pathlib import Path


def audit(root):
    records = []
    for seed in sorted(Path(root).glob("seed-*")):
        if not (seed / "epochs.csv").exists():
            continue
        epochs = list(csv.DictReader((seed / "epochs.csv").open()))
        complete = [r for r in epochs if r.get("epoch") and r.get("epoch_wall_seconds")]
        selected = [r for r in complete if 2 <= int(r["epoch"]) <= 20]
        if len(selected) != 19:
            continue
        step_path = seed / "steps.csv"
        if not step_path.exists():
            step_path = seed / "steps.csv.gz"
        sums = {
            key: 0.0 for key in ("read_seconds", "h2d_seconds", "compute_seconds", "step_seconds")
        }
        steps = 0
        opener = gzip.open if step_path.suffix == ".gz" else open
        with opener(step_path, "rt") as f:
            for row in csv.DictReader(f):
                epoch = int(row["epoch"])
                if epoch > 20:
                    break
                if epoch < 2:
                    continue
                steps += 1
                for k in sums:
                    sums[k] += float(row[k])
        assert steps == 19 * 342, (str(seed), steps)
        config = json.loads((seed / "configuration.json").read_text())
        record = dict(
            seed=seed.name,
            completed_epoch=max(int(r["epoch"]) for r in complete),
            steps=steps,
            step_mean_ms={k: 1000 * v / steps for k, v in sums.items()},
            mean_train_epoch_seconds=st.mean(float(r["train_seconds"]) for r in selected),
            mean_eval_epoch_seconds=st.mean(
                sum(
                    float(r[k])
                    for k in [
                        "train_evaluation_seconds",
                        "validation_evaluation_seconds",
                        "test_evaluation_seconds",
                    ]
                )
                for r in selected
            ),
            mean_epoch_wall_seconds=st.mean(float(r["epoch_wall_seconds"]) for r in selected),
            epoch20_validation_pearson_delta=float(selected[-1]["validation_pearson_delta"]),
            config={
                k: config.get(k)
                for k in [
                    "batch_size",
                    "gradient_accumulation",
                    "precision",
                    "data_manifest_sha256",
                    "parameter_count",
                    "optimizer",
                    "learning_rate",
                    "training_rows",
                ]
            },
        )
        records.append(record)
    return records


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1])))
