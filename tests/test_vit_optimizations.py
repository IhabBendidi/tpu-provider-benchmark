import copy

import torch
from vit_optimizations import (
    FLAT_ALIGNMENT,
    AlignedFlatAdamW,
    FunctionalAdamW,
    WholeStepRunner,
    apply_attention,
)


def _small_vit_loss_model():
    from workload import LossModel

    from gpu_pilot.config import ImagingSettings
    from gpu_pilot.imaging import VisionTransformer

    settings = ImagingSettings(
        profile="local_smoke",
        dataset_size=64,
        validation_sample_count=16,
        class_count=6,
        compound_count=20,
        train_compound_count=16,
        image_size=32,
        patch_size=16,
        embedding_size=48,
        transformer_depth=2,
        attention_heads=4,
        mlp_size=96,
        global_batch_size=16,
        micro_batch_size_per_rank=4,
        expected_world_size=1,
        learning_rate=0.001,
        weight_decay=0.01,
    )
    torch.manual_seed(17)
    return LossModel(VisionTransformer(settings), "synthetic_imaging_vit")


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


def test_aligned_flat_adamw_128_alignment_and_dtype_partitioning():
    model = _small_vit_loss_model()
    ref_params = [torch.nn.Parameter(p.detach().clone()) for p in model.parameters()]
    reference = torch.optim.AdamW(
        ref_params, lr=0.001, weight_decay=0.01, foreach=False, fused=False
    )
    flat_opt = AlignedFlatAdamW(
        model,
        lr=0.001,
        weight_decay=0.01,
        precision="bf16",
        compile_update=False,
        world_size=4,
        reduce_grads=True,
    )

    assert flat_opt.total_aligned_numel % FLAT_ALIGNMENT == 0
    for (name, _), (offset, numel, _, pad_len, target_dtype) in zip(
        model.named_parameters(), flat_opt.specs, strict=True
    ):
        assert offset % FLAT_ALIGNMENT == 0
        assert (numel + pad_len) % FLAT_ALIGNMENT == 0
        if any(k in name for k in ("class_token", "position_embedding", ".norm")):
            assert target_dtype == torch.float32
        else:
            assert target_dtype == torch.bfloat16

    initial_working = flat_opt.initial_working_params()
    assert all(t._base is None and not t.requires_grad for t in initial_working)

    for _ in range(12):
        prev_param_ref = flat_opt.flat_param
        prev_param_snapshot = prev_param_ref.clone()
        prev_step_ref = flat_opt.step_tensor
        prev_step_val = float(prev_step_ref.item())
        grads_for_flat = []
        for p in ref_params:
            avg_grad = torch.randn_like(p)
            p.grad = avg_grad.clone()
            grads_for_flat.append(avg_grad * 4.0)
        reference.step()
        flat_grad = flat_opt.pack_gradients(grads_for_flat)
        emitted = flat_opt._update_and_emit(flat_grad)
        assert flat_opt.flat_param.data_ptr() != prev_param_ref.data_ptr()
        torch.testing.assert_close(prev_param_ref, prev_param_snapshot, atol=0.0, rtol=0.0)
        assert float(prev_step_ref.item()) == prev_step_val
        assert all(t._base is None and not t.requires_grad for t in emitted)

    for idx, p in enumerate(ref_params):
        torch.testing.assert_close(p, flat_opt.master_params[idx], atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(reference.state[p]["exp_avg"], flat_opt.m[idx])
        torch.testing.assert_close(reference.state[p]["exp_avg_sq"], flat_opt.v[idx])
    assert flat_opt.step_tensor.item() == 12


def test_whole_step_runner_vit_matches_four_microbatch_reference():
    ref_model = _small_vit_loss_model()
    apply_attention(ref_model)
    opt_model = copy.deepcopy(ref_model)
    opt_model_repeat = copy.deepcopy(ref_model)
    reference_opt = torch.optim.AdamW(
        ref_model.parameters(), lr=0.001, weight_decay=0.01, foreach=False, fused=False
    )

    torch.manual_seed(23)
    microbatches = [
        (torch.randn(4, 3, 32, 32), torch.randint(0, 6, (4,))) for _ in range(4)
    ]
    flat_inputs = tuple(tensor for pair in microbatches for tensor in pair)
    runner = WholeStepRunner(
        model=opt_model,
        accumulation=4,
        lr=0.001,
        weight_decay=0.01,
        precision="fp32",
        example_flat_inputs=flat_inputs,
        compile_step=False,
        trace_fx=True,
    )

    first_step_losses = []
    for _ in range(5):
        reference_opt.zero_grad(set_to_none=True)
        ref_loss_value = 0.0
        for images, labels in microbatches:
            loss = ref_model(images, labels)
            (loss / 4.0).backward()
            ref_loss_value += float(loss.detach()) / 4.0
        reference_opt.step()

        step_loss = float(runner.step(*flat_inputs))
        first_step_losses.append(step_loss)
        assert abs(step_loss - ref_loss_value) < 1e-5

    runner.sync_model_parameters()
    for p_ref, p_opt in zip(ref_model.parameters(), opt_model.parameters(), strict=True):
        torch.testing.assert_close(p_ref, p_opt, atol=1e-5, rtol=1e-5)

    # Second runner with identical architecture/shapes hits _COMPILED_V6E_CACHE
    # while resetting weights and optimizer moments cleanly.
    runner_repeat = WholeStepRunner(
        model=opt_model_repeat,
        accumulation=4,
        lr=0.001,
        weight_decay=0.01,
        precision="fp32",
        example_flat_inputs=flat_inputs,
        compile_step=False,
        trace_fx=True,
    )
    assert runner_repeat.cache_hit is True
    assert runner_repeat.optimizer.step_tensor.item() == 0
    repeat_losses = [float(runner_repeat.step(*flat_inputs)) for _ in range(5)]
    assert repeat_losses == first_step_losses

    bf16_model = _small_vit_loss_model()
    bf16_runner = WholeStepRunner(
        model=bf16_model,
        accumulation=4,
        lr=0.001,
        weight_decay=0.01,
        precision="bf16",
        example_flat_inputs=flat_inputs,
        compile_step=False,
        trace_fx=True,
    )
    bf16_loss = float(bf16_runner.step(*flat_inputs))
    assert torch.isfinite(torch.tensor(bf16_loss))


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
    ref = run_phase(baseline_args, config, backend, "fixed_job", updates=8)
    opt1 = run_phase(opt_args, config, backend, "fixed_job", updates=8)
    opt2 = run_phase(opt_args, config, backend, "fixed_job_repeat", updates=8)
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
