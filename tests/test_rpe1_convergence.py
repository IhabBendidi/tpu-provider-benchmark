import argparse
import csv
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from gpu_pilot import rpe1_data
from gpu_pilot.rpe1_data import csv_write, digest, make_splits, normalize_selected
from gpu_pilot.rpe1_train import PerturbationMlp, initialize, pearson_rows, plateau_status, train


def test_plateau_requires_both_validation_metrics_and_minimum_epochs():
    rows = [
        {
            "epoch": e,
            "validation_mse": 0.1,
            "validation_pearson_delta": 0.6,
            "test_pearson_delta": e / 1000,
        }
        for e in range(101)
    ]
    assert not plateau_status(rows[:100])["plateau"]
    assert plateau_status(rows)["plateau"]
    # Recent improvement in either metric prevents stopping; test scores do not.
    for metric, value in (("validation_mse", 0.099), ("validation_pearson_delta", 0.61)):
        changed = [dict(r) for r in rows]
        for row in changed[-10:]:
            row[metric] = value
        assert not plateau_status(changed)["plateau"]


def test_plateau_accumulates_small_improvements_and_ignores_single_epoch_noise():
    rows = [
        {"epoch": e, "validation_mse": 0.1 - (e // 10) * 5e-6, "validation_pearson_delta": 0.6}
        for e in range(101)
    ]
    assert not plateau_status(rows)["plateau"]
    flat = [
        {"epoch": e, "validation_mse": 0.1, "validation_pearson_delta": 0.6} for e in range(101)
    ]
    flat[-1]["validation_pearson_delta"] += 0.005
    assert plateau_status(flat)["plateau"]


def test_replay_verification_detects_changed_parameters(tmp_path):
    from gpu_pilot.rpe1_train import save_weights, verify_weights

    model = PerturbationMlp(4, 10)
    initialize(model, 7)
    checkpoint = tmp_path / "weights.npz"
    save_weights(checkpoint, model)
    assert all(r["exact_equal"] for r in verify_weights(checkpoint, model))
    with torch.no_grad():
        model.encoder[0].weight[0, 0] += 0.001
    assert not all(r["exact_equal"] for r in verify_weights(checkpoint, model))


def test_support_query_splits_are_reproducible_disjoint_and_target_grouped():
    labels = np.r_[np.full(100, -1), np.repeat(np.arange(20), 30)]
    parts, cohorts, support = make_splits(labels, 17)
    again = make_splits(labels, 17)
    assert all(np.array_equal(a, b) for a, b in zip((parts, cohorts, support), again, strict=True))
    assert len(set(np.flatnonzero(parts == 0)) & set(np.flatnonzero(parts != 0))) == 0
    assert set(labels[(parts == 0) & (labels >= 0)]) == set(range(20))
    assert sum(cohorts[labels == p][0] == 2 for p in range(20)) == 4
    assert sum(cohorts[labels == p][0] == 1 for p in range(20)) == 2
    for p in range(20):
        role = cohorts[labels == p][0]
        assert len(set(cohorts[labels == p])) == 1
        assert sum(support[labels == p]) == (3 if role else 0)
        if role:
            assert sum((parts == role) & (labels == p)) == 27
    assert [sum((labels == -1) & (parts == p)) for p in range(3)] == [70, 10, 20]


def test_normalization_is_library_3000_then_log1p_without_quantization():
    out = normalize_selected(np.array([[1, 3], [0, 5]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.expm1(out), [[750, 2250], [0, 3000]], rtol=1e-6)
    with pytest.raises(ValueError):
        normalize_selected(np.zeros((1, 2)))


def test_pearson_delta_is_across_genes_and_undefined_vectors_are_explicit():
    baseline = np.array([10, 20, 30, 40])
    truth = np.array([[1, 3, 2, 4], [1, 3, 2, 4], [0, 0, 0, 0]])
    pred = np.array([[2, 6, 4, 8], [-1, -3, -2, -4], [0, 0, 0, 0]])
    score = pearson_rows((pred + baseline) - baseline, (truth + baseline) - baseline)
    np.testing.assert_allclose(score[:2], [1, -1], atol=1e-12)
    assert np.isnan(score[2])


def test_initialization_and_one_hot_conditioning_are_portable():
    a, b = PerturbationMlp(4, 10), PerturbationMlp(4, 10)
    initialize(a, 123)
    initialize(b, 123)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    controls = torch.ones((2, 4))
    prediction = a(controls, torch.tensor([0, 1]))
    assert prediction.shape == (2, 4)
    assert not torch.equal(prediction[0], prediction[1])


def test_cpu_end_to_end_has_epoch_zero_all_metrics_and_success(tmp_path):
    package, local, output = [tmp_path / p for p in ("package", "local", "output")]
    package.mkdir()
    local.mkdir()
    labels = np.r_[np.full(30, -1), np.repeat(np.arange(10), 20)].astype(np.int32)
    parts, cohorts, _ = make_splits(labels, 42)
    rng = np.random.default_rng(42)
    x = normalize_selected(rng.poisson(4, (len(labels), 4)) + 1)
    paired = np.empty(len(labels), dtype=np.int64)
    for p in range(3):
        rows = np.flatnonzero(parts == p)
        paired[rows] = rng.choice(np.flatnonzero((parts == p) & (labels == -1)), len(rows))
    for name, value in dict(
        expression=x,
        labels=labels,
        partitions=parts,
        eval_control=paired,
        control_mean=x[(parts == 0) & (labels == -1)].mean(axis=0),
    ).items():
        np.save(local / f"{name}.npy", value)
    (package / "manifest.json").write_text(
        json.dumps({"genes": 4, "perturbations": 10, "split_seed": 42})
    )
    (local / "_SUCCESS").write_text(digest(package / "manifest.json"))
    csv_write(
        package / "perturbations.csv",
        [{"perturbation": str(p), "cohort": int(cohorts[labels == p][0])} for p in range(10)],
    )
    args = argparse.Namespace(
        package=package,
        local=local,
        output=output,
        phase="unit_test",
        seed=3,
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        selection=None,
        device="cpu",
        max_train_steps=2,
        max_eval_cells=0,
    )
    train(args)
    rows = list(csv.DictReader((output / "epochs.csv").open()))
    assert [r["epoch"] for r in rows] == ["0", "1"]
    for row in rows:
        for split in ("train", "validation", "test"):
            assert np.isfinite(float(row[f"{split}_mse"]))
            assert np.isfinite(float(row[f"{split}_pearson_delta"]))
    assert json.loads((output / "_SUCCESS").read_text())["status"] == "complete"


def test_preprocessing_ignores_query_expression_for_gene_selection_and_stages_exactly(
    tmp_path, monkeypatch
):
    labels = np.r_[np.full(30, -1), np.repeat(np.arange(10), 20)].astype(np.int32)
    parts, _, _ = make_splits(labels, 42)
    counts = np.ones((len(labels), 4), dtype=np.float32)
    counts[:, 0] = 100
    counts[:, 1] = 50
    counts[parts != 0, 3] = 1000000
    source = tmp_path / "raw.h5ad"
    with h5py.File(source, "w") as f:
        x = f.create_dataset("X", data=counts)
        x.attrs["encoding-type"] = "array"
        obs = f.create_group("obs")
        obs.attrs["_index"] = "cell_id"
        obs.create_dataset(
            "cell_id", data=np.asarray([f"cell-{i}" for i in range(len(labels))], dtype="S")
        )
        obs.create_dataset(
            "perturbation",
            data=np.asarray(["control" if p < 0 else f"G{p}" for p in labels], dtype="S"),
        )
        var = f.create_group("var")
        var.attrs["_index"] = "gene_name"
        var.create_dataset("gene_name", data=np.asarray(["a", "b", "c", "d"], dtype="S"))
    monkeypatch.setattr(rpe1_data, "SOURCE_MD5", digest(source, "md5"))
    # The test filesystem is RAM-backed and smaller than the production disk budget.
    monkeypatch.setattr(
        rpe1_data.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=3 * 1024**3),
    )
    package, first, second = [tmp_path / name for name in ("package", "first", "second")]
    rpe1_data.prepare(source, package, first, 42, genes_count=2)
    genes = list(csv.DictReader((package / "genes.csv").open()))
    assert [g["gene_symbol"] for g in genes] == ["a", "b"]
    rpe1_data.stage(package, second)
    assert digest(first / "expression.npy") == digest(second / "expression.npy")
    metadata = np.load(package / "metadata.npz")
    assert np.all(metadata["partitions"][metadata["eval_control"]] == metadata["partitions"])
    assert np.all(metadata["labels"][metadata["eval_control"]] == -1)
