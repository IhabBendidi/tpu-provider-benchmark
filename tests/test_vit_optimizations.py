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
