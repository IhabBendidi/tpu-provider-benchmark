import copy

import torch
from vit_optimizations import FunctionalAdamW, apply_attention


def test_functional_adamw_preserves_updates_and_state():
    torch.manual_seed(8)
    parameters = [torch.nn.Parameter(torch.randn(7, 4)), torch.nn.Parameter(torch.randn(7))]
    candidates = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    reference = torch.optim.AdamW(
        parameters, lr=0.001, weight_decay=0.01, foreach=False, fused=False
    )
    adapted = FunctionalAdamW(candidates, lr=0.001, weight_decay=0.01, compile_update=False)
    for _ in range(30):
        for p, q in zip(parameters, candidates, strict=True):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        reference.step()
        adapted.step()
    for index, (p, q) in enumerate(zip(parameters, candidates, strict=True)):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(reference.state[p]["exp_avg"], adapted.m[index])
        torch.testing.assert_close(reference.state[p]["exp_avg_sq"], adapted.v[index])
    assert adapted.step_tensor.item() == 30


def test_functional_adamw_fused_world_size_scaling():
    torch.manual_seed(11)
    parameters = [torch.nn.Parameter(torch.randn(7, 4)), torch.nn.Parameter(torch.randn(7))]
    candidates = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    reference = torch.optim.AdamW(
        parameters, lr=0.001, weight_decay=0.01, foreach=False, fused=False
    )
    adapted = FunctionalAdamW(
        candidates,
        lr=0.001,
        weight_decay=0.01,
        compile_update=False,
        world_size=4,
        reduce_grads=True,
    )
    for _ in range(15):
        for p, q in zip(parameters, candidates, strict=True):
            avg_grad = torch.randn_like(p)
            p.grad = avg_grad.clone()
            q.grad = (avg_grad * 4.0).clone()
        reference.step()
        adapted._update([q.grad for q in candidates])
    for index, (p, q) in enumerate(zip(parameters, candidates, strict=True)):
        torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(reference.state[p]["exp_avg"], adapted.m[index])
        torch.testing.assert_close(reference.state[p]["exp_avg_sq"], adapted.v[index])
    assert adapted.step_tensor.item() == 15


def test_v6e_opt_phase_reset_and_loss_parity(tmp_path):
    import argparse

    import numpy as np
    from backend import Backend
    from historical_benchmark import run_phase
    from workload import make_model

    from gpu_pilot.config import load_config

    tpu_root = copy.deepcopy(
        __import__("pathlib").Path(__file__).resolve().parents[1] / "src/tpu_benchmark"
    )
    config = load_config(tpu_root / "baseline/config/benchmark.synthetic.local.toml")
    config.run.mode = "gpu_only"
    state = make_model(config).state_dict()
    fixture = tmp_path / "initial.npz"
    np.savez(fixture, **{k: v.numpy() for k, v in state.items()})
    np.save(fixture.with_suffix(".rng.npy"), torch.get_rng_state().numpy())
    backend = Backend("cpu-test")
    torch.set_num_threads(1)

    baseline_args = argparse.Namespace(
        initial_state=fixture, communication="per-parameter", optimization="baseline"
    )
    opt_args = argparse.Namespace(
        initial_state=fixture, communication="per-parameter", optimization="v6e_opt"
    )
    ref = run_phase(baseline_args, config, backend, "fixed_job", updates=4)
    opt1 = run_phase(opt_args, config, backend, "fixed_job", updates=4)
    opt2 = run_phase(opt_args, config, backend, "fixed_job_repeat", updates=4)
    assert opt1["samples_processed"] == ref["samples_processed"]
    assert opt1["first_training_loss"] == opt2["first_training_loss"]
    assert opt1["last_training_loss"] == opt2["last_training_loss"]
    assert opt1["validation_loss"] == opt2["validation_loss"]
    assert abs(opt1["first_training_loss"] - ref["first_training_loss"]) < 1e-5
    assert abs(opt1["last_training_loss"] - ref["last_training_loss"]) < 1e-5
    assert abs(opt1["validation_loss"] - ref["validation_loss"]) < 1e-5


def test_sdpa_preserves_attention_and_all_gradients():
    torch.manual_seed(9)
    original = torch.nn.MultiheadAttention(48, 4, batch_first=True)
    adapted = copy.deepcopy(original)
    apply_attention(adapted)
    x = torch.randn(2, 17, 48, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    a = original(x, x, x, need_weights=False)[0]
    b = adapted(y, y, y, need_weights=False)[0]
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.square().mean().backward()
    b.square().mean().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-7, rtol=1e-4)
    for p, q in zip(original.parameters(), adapted.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-4)
