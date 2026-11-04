"""Architecture-preserving ViT execution variants for the single-trial campaign."""

import torch
from torch import nn
from torch.nn import functional as F


class FunctionalAdamW:
    """AdamW update compiled as one graph, with FP32 state and original equations."""

    def __init__(
        self,
        parameters,
        lr,
        weight_decay,
        compile_update=True,
        world_size=1,
        reduce_grads=False,
    ):
        self.params = list(parameters)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.step_tensor = torch.zeros((), device=self.params[0].device, dtype=torch.float32)
        self.lr, self.weight_decay = lr, weight_decay
        self.world_size = int(world_size)
        self.reduce_grads = bool(reduce_grads and self.world_size > 1)
        self.grad_scale = 1.0 / self.world_size if self.reduce_grads else 1.0
        self.update = (
            torch.compile(self._update, backend="tpu", fullgraph=True)
            if compile_update
            else self._update
        )

    @torch.no_grad()
    def _update(self, gradients):
        self.step_tensor.add_(1)
        step_size = -(self.lr / (1.0 - 0.9**self.step_tensor))
        inv_corr2_sqrt = (1.0 - 0.999**self.step_tensor).rsqrt()
        decay = 1.0 - self.lr * self.weight_decay
        grad_scale = self.grad_scale
        for p, g, m, v in zip(self.params, gradients, self.m, self.v, strict=True):
            if grad_scale != 1.0:
                g = g * grad_scale
            p.mul_(decay)
            m.lerp_(g, 0.1)
            v.mul_(0.999).addcmul_(g, g, value=0.001)
            denominator = v.sqrt().mul_(inv_corr2_sqrt).add_(1e-8)
            p.add_(step_size * m / denominator)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def step(self):
        gradients = [
            p.grad.clone() if p.grad is not None and p.grad._base is not None else p.grad
            for p in self.params
        ]
        if any(g is None for g in gradients):
            raise ValueError("Functional AdamW requires a gradient for every ViT parameter")
        if self.reduce_grads:
            for g in gradients:
                torch.distributed.all_reduce(g, op=torch.distributed.ReduceOp.SUM)
        self.update(gradients)


def sdpa_forward(self, query, key, value, **kwargs):
    """Exact self-attention for this ViT; preserve projection tensors/state names."""
    if query is not key or key is not value or kwargs.get("need_weights", True):
        raise ValueError("Variant is restricted to ViT self-attention without returned weights")
    if kwargs.get("attn_mask") is not None or kwargs.get("key_padding_mask") is not None:
        raise ValueError("Masked attention requires separate correctness validation")
    b, length, width = query.shape
    head_dim = width // self.num_heads
    bh = b * self.num_heads
    scale = head_dim**-0.5
    q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
    q = q.reshape(b, length, self.num_heads, head_dim).transpose(1, 2).reshape(bh, length, head_dim)
    k = k.reshape(b, length, self.num_heads, head_dim).transpose(1, 2).reshape(bh, length, head_dim)
    v = v.reshape(b, length, self.num_heads, head_dim).transpose(1, 2).reshape(bh, length, head_dim)
    attn = torch.bmm(q, k.transpose(1, 2)) * scale
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
    output = torch.bmm(attn, v).reshape(b, self.num_heads, length, head_dim)
    output = output.transpose(1, 2).reshape(b, length, width)
    return F.linear(output, self.out_proj.weight, self.out_proj.bias), None


def apply_attention(model):
    import types

    count = 0
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            if (
                module.dropout
                or not module.batch_first
                or module.bias_k is not None
                or module.add_zero_attn
            ):
                raise ValueError("Unsupported attention configuration")
            module.forward = types.MethodType(sdpa_forward, module)
            count += 1
    if not count:
        raise ValueError("No ViT attention modules found")
