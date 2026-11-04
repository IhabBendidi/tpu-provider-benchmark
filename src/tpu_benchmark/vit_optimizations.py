"""Architecture-preserving ViT execution variants for the single-trial campaign."""

import torch
from torch import nn
from torch.nn import functional as F


class FunctionalAdamW:
    """AdamW update compiled as one graph, with FP32 state and original equations."""

    def __init__(self, parameters, lr, weight_decay, compile_update=True):
        self.params = list(parameters)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.step_tensor = torch.zeros((), device=self.params[0].device, dtype=torch.float32)
        self.lr, self.weight_decay = lr, weight_decay
        self.update = (
            torch.compile(self._update, backend="tpu", fullgraph=True)
            if compile_update
            else self._update
        )

    @torch.no_grad()
    def _update(self, gradients):
        self.step_tensor.add_(1)
        correction1 = 1 - 0.9**self.step_tensor
        correction2_sqrt = (1 - 0.999**self.step_tensor).sqrt()
        for p, g, m, v in zip(self.params, gradients, self.m, self.v, strict=True):
            p.mul_(1 - self.lr * self.weight_decay)
            m.lerp_(g, 0.1)
            v.mul_(0.999).addcmul_(g, g, value=0.001)
            denominator = v.sqrt() / correction2_sqrt + 1e-8
            p.add_(-(self.lr / correction1) * m / denominator)

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def step(self):
        gradients = [p.grad for p in self.params]
        if any(g is None for g in gradients):
            raise ValueError("Functional AdamW requires a gradient for every ViT parameter")
        self.update(gradients)


def sdpa_forward(self, query, key, value, **kwargs):
    """Exact self-attention for this ViT; preserve projection tensors/state names."""
    if query is not key or key is not value or kwargs.get("need_weights", True):
        raise ValueError("Variant is restricted to ViT self-attention without returned weights")
    if kwargs.get("attn_mask") is not None or kwargs.get("key_padding_mask") is not None:
        raise ValueError("Masked attention requires separate correctness validation")
    b, length, width = query.shape
    q, k, v = F.linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)

    def heads(x):
        return (
            x.reshape(b, length, self.num_heads, width // self.num_heads)
            .transpose(1, 2)
            .contiguous()
        )

    output = F.scaled_dot_product_attention(
        heads(q), heads(k), heads(v), dropout_p=0.0, is_causal=False
    )
    output = output.transpose(1, 2).contiguous().reshape(b, length, width)
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
