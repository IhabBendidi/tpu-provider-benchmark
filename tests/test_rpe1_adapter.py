"""Check the TPU port against the original CPU convergence protocol."""

import csv
import importlib.util
import json
from pathlib import Path

import pytest
import torch
from rpe1_train import train


def test_rpe1_cpu_port_preserves_original_learning_curve(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parent / "test_rpe1_convergence.py"
    spec = importlib.util.spec_from_file_location("original_rpe1_tests", path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    reference, adapted = tmp_path / "reference", tmp_path / "adapted"
    reference.mkdir()
    adapted.mkdir()
    original.test_cpu_end_to_end_has_epoch_zero_all_metrics_and_success(reference)
    # Inspect real forward calls: evaluation must disable gradients without
    # creating inference tensors; training must still have gradients enabled.
    forward = original.PerturbationMlp.forward
    modes = []

    def checked_forward(self, *args, **kwargs):
        modes.append((self.training, torch.is_grad_enabled(), torch.is_inference_mode_enabled()))
        return forward(self, *args, **kwargs)

    monkeypatch.setattr(original.PerturbationMlp, "forward", checked_forward)
    monkeypatch.setattr(original, "train", train)
    original.test_cpu_end_to_end_has_epoch_zero_all_metrics_and_success(adapted)
    left = list(csv.DictReader((reference / "output/epochs.csv").open()))
    right = list(csv.DictReader((adapted / "output/epochs.csv").open()))
    for a, b in zip(left, right, strict=True):
        for field in (
            "optimizer_updates",
            "train_mse",
            "validation_mse",
            "test_mse",
            "train_pearson_delta",
            "validation_pearson_delta",
            "test_pearson_delta",
        ):
            assert float(a[field]) == pytest.approx(float(b[field]), abs=1e-9)
    config = json.loads((adapted / "output/configuration.json").read_text())
    assert config["precision"] == "float32"
    assert config["matmul_precision"] == "highest"
    assert config["independent_seed_no_ddp"]

    assert any(training and grad for training, grad, _ in modes)
    assert any(not training for training, _, _ in modes)
    assert all(not grad and not inference for training, grad, inference in modes if not training)
