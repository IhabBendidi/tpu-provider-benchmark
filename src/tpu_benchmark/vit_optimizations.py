"""Architecture-preserving ViT execution variants for the single-trial campaign."""

import math
import os
import types
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class FunctionalAdamW:
    """AdamW update compiled as one graph, with FP32 state and original equations."""

    def __init__(self, parameters, lr, weight_decay, compile_update=True, world_size=1):
        self.params = list(parameters)
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.step_tensor = torch.zeros((), device=self.params[0].device, dtype=torch.float32)
        self.lr, self.weight_decay = lr, weight_decay
        self.world_size = world_size
        self.update = (
            torch.compile(self._update, backend="tpu", fullgraph=True)
            if compile_update
            else self._update
        )
        self.fused_accum_reduce_update = (
            torch.compile(self._fused_accum_reduce_update, backend="tpu", fullgraph=True)
            if compile_update
            else self._fused_accum_reduce_update
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

    @torch.no_grad()
    def _fused_accum_reduce_update(self, micro_gradients, micro_losses):
        accumulation = len(micro_gradients)
        scale = 1.0 / (accumulation * self.world_size)
        self.step_tensor.add_(1)
        correction1 = 1 - 0.9**self.step_tensor
        correction2_sqrt = (1 - 0.999**self.step_tensor).sqrt()
        step_size = -(self.lr / correction1)
        for idx, (p, m, v) in enumerate(zip(self.params, self.m, self.v, strict=True)):
            g = micro_gradients[0][idx]
            for micro in range(1, accumulation):
                g = g + micro_gradients[micro][idx]
            if self.world_size > 1:
                torch.distributed.all_reduce(g)
            g = g * scale
            p.mul_(1 - self.lr * self.weight_decay)
            m.lerp_(g, 0.1)
            v.mul_(0.999).addcmul_(g, g, value=0.001)
            denominator = v.sqrt() / correction2_sqrt + 1e-8
            p.add_((step_size * m) / denominator)
        total_loss = micro_losses[0]
        for micro in range(1, len(micro_losses)):
            total_loss = total_loss + micro_losses[micro]
        return total_loss * (1.0 / len(micro_losses))

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

    def step_accumulated(self, micro_gradients, micro_losses):
        return self.fused_accum_reduce_update(micro_gradients, micro_losses)


def linear_3d_forward(self, x):
    """3D batched matmul projection avoiding unaligned 2D tile flatten/unflatten."""
    out = torch.matmul(x, self.weight.t())
    if self.bias is not None:
        if self.out_features == 3072 and os.environ.get("VIT_OPT_CHUNK_QKV", "1") != "1":
            out = (out + self.bias).to(out.dtype)
        else:
            out = out + self.bias
    return out


def sdpa_forward(self, query, key, value, **kwargs):
    """Exact self-attention for this ViT; preserve projection tensors/state names."""
    if query is not key or key is not value or kwargs.get("need_weights", True):
        raise ValueError("Variant is restricted to ViT self-attention without returned weights")
    if kwargs.get("attn_mask") is not None or kwargs.get("key_padding_mask") is not None:
        raise ValueError("Masked attention requires separate correctness validation")
    b, length, width = query.shape
    head_dim = width // self.num_heads
    scale = 1.0 / math.sqrt(head_dim)
    qkv = torch.matmul(query, self.in_proj_weight.t())
    if os.environ.get("VIT_OPT_CHUNK_QKV", "1") == "1":
        q, k, v = qkv.chunk(3, dim=-1)
        if self.in_proj_bias is not None:
            qb, kb, vb = self.in_proj_bias.chunk(3, dim=0)
            q = q + qb
            k = k + kb
            v = v + vb
        q = q.view(b, length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.view(b, length, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.view(b, length, self.num_heads, head_dim).permute(0, 2, 1, 3)
    else:
        if self.in_proj_bias is not None:
            qkv = qkv + self.in_proj_bias
        qkv = qkv.view(b, length, 3, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
    attn = torch.matmul(q * scale, k.transpose(-2, -1))
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).reshape(b, length, width)
    out = torch.matmul(output, self.out_proj.weight.t())
    if self.out_proj.bias is not None:
        out = out + self.out_proj.bias
    return out, None


def checkpointed_block_forward(self, values):
    """Rematerialize intermediate block activations to eliminate HBM attention spills."""
    def _block_body(v):
        normalized = self.norm1(v)
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        v = v + attended
        return v + self.mlp(self.norm2(v))

    if self.training and torch.is_grad_enabled():
        return checkpoint(_block_body, values, use_reentrant=False)
    return _block_body(values)


def apply_attention(model):
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
        elif isinstance(module, nn.Linear):
            module.forward = types.MethodType(linear_3d_forward, module)
        elif module.__class__.__name__ == "TransformerBlock":
            module.forward = types.MethodType(checkpointed_block_forward, module)
    if not count:
        raise ValueError("No ViT attention modules found")
