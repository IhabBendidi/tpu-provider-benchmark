"""Architecture-preserving ViT execution variants for the single-trial campaign."""

import os

import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
from torch.nn import functional as F
from workload import masked_mse

FLAT_ALIGNMENT = 128
_COMPILED_V6E_CACHE: dict[tuple, tuple] = {}


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


def resolve_working_dtypes(model: nn.Module, precision: str) -> list[torch.dtype]:
    """Select BF16 working weights for Conv2d/Linear/MHA while keeping norms/pos_embed in FP32."""
    modules_by_name = dict(model.named_modules())
    use_bf16 = precision == "bf16"
    dtypes: list[torch.dtype] = []
    for full_name, param in model.named_parameters():
        if not use_bf16:
            dtypes.append(torch.float32)
            continue
        parent_name = full_name.rsplit(".", 1)[0] if "." in full_name else ""
        owner = modules_by_name[parent_name]
        if isinstance(owner, nn.Conv2d | nn.Linear | nn.MultiheadAttention):
            dtypes.append(torch.bfloat16)
        else:
            dtypes.append(param.dtype)
    return dtypes


def _functional_adamw_step_and_emit(
    step_tensor: torch.Tensor,
    flat_param: torch.Tensor,
    flat_m: torch.Tensor,
    flat_v: torch.Tensor,
    scaled_flat_grad: torch.Tensor,
    lr: float,
    weight_decay: float,
    specs: tuple[tuple[int, int, tuple[int, ...], int, torch.dtype], ...],
) -> tuple[torch.Tensor, ...]:
    """Purely out-of-place 1D FP32 AdamW update + BF16/FP32 working-weight emission."""
    new_step = step_tensor + 1.0
    step_size = -(lr / (1.0 - 0.9**new_step))
    inv_corr2_sqrt = (1.0 - 0.999**new_step).rsqrt()
    decay = 1.0 - lr * weight_decay
    decayed_param = flat_param * decay
    new_flat_m = torch.lerp(flat_m, scaled_flat_grad, 0.1)
    new_flat_v = torch.addcmul(
        flat_v * 0.999, scaled_flat_grad, scaled_flat_grad, value=0.001
    )
    denominator = new_flat_v.sqrt() * inv_corr2_sqrt + 1e-8
    new_flat_param = decayed_param + step_size * new_flat_m / denominator
    outputs: list[torch.Tensor] = [new_step, new_flat_param, new_flat_m, new_flat_v]
    for offset, numel, shape, _, target_dtype in specs:
        sliced = new_flat_param[offset : offset + numel].reshape(shape)
        if target_dtype != sliced.dtype:
            outputs.append(sliced.to(target_dtype))
        else:
            outputs.append(sliced.clone())
    return tuple(outputs)


class AlignedFlatAdamW:
    """128-element-aligned 1D FP32 master AdamW emitting pre-cast BF16/FP32 working weights."""

    def __init__(
        self,
        model_or_params,
        lr: float,
        weight_decay: float,
        precision: str = "bf16",
        working_dtypes=None,
        compile_update: bool = True,
        world_size: int = 1,
        reduce_grads: bool = False,
        alignment: int = FLAT_ALIGNMENT,
    ):
        if isinstance(model_or_params, nn.Module):
            self.params = list(model_or_params.parameters())
            if working_dtypes is None:
                working_dtypes = resolve_working_dtypes(model_or_params, precision)
        else:
            self.params = list(model_or_params)
            if working_dtypes is None:
                default_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
                working_dtypes = [default_dtype for _ in self.params]
        if not self.params:
            raise ValueError("AlignedFlatAdamW requires at least one parameter")
        if alignment <= 0 or alignment % 128 != 0:
            raise ValueError("alignment must be a positive multiple of 128 elements")
        self.alignment = int(alignment)
        self.working_dtypes = list(working_dtypes)
        if len(self.working_dtypes) != len(self.params):
            raise ValueError("working_dtypes length must match parameter count")

        device = self.params[0].device
        self.zero_pad = torch.zeros(self.alignment, device=device, dtype=torch.float32)
        cpu_zero_pad = torch.zeros(self.alignment, dtype=torch.float32)
        specs_list: list[tuple[int, int, tuple[int, ...], int, torch.dtype]] = []
        pieces: list[torch.Tensor] = []
        cursor = 0
        for p, target_dtype in zip(self.params, self.working_dtypes, strict=True):
            numel = p.numel()
            aligned_numel = ((numel + self.alignment - 1) // self.alignment) * self.alignment
            pad_len = aligned_numel - numel
            specs_list.append((cursor, numel, tuple(p.shape), pad_len, target_dtype))
            pieces.append(p.detach().cpu().float().reshape(-1))
            if pad_len:
                pieces.append(cpu_zero_pad[:pad_len])
            cursor += aligned_numel

        self.specs = tuple(specs_list)
        self.total_aligned_numel = cursor
        self.pad_lengths = tuple(spec[3] for spec in self.specs)
        self.flat_param = torch.cat(pieces, dim=0).contiguous().to(device)
        self.flat_m = torch.zeros_like(self.flat_param)
        self.flat_v = torch.zeros_like(self.flat_param)
        self.step_tensor = torch.zeros((), device=device, dtype=torch.float32)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.world_size = int(world_size)
        self.reduce_grads = bool(reduce_grads and self.world_size > 1)
        self.grad_scale = 1.0 / self.world_size if self.reduce_grads else 1.0

        lr_val = self.lr
        wd_val = self.weight_decay
        grad_scale = self.grad_scale
        specs = self.specs

        def _pure_single_update(step_tensor, flat_param, flat_m, flat_v, flat_grad):
            scaled = flat_grad * grad_scale if grad_scale != 1.0 else flat_grad
            return _functional_adamw_step_and_emit(
                step_tensor, flat_param, flat_m, flat_v, scaled, lr_val, wd_val, specs
            )

        self._compiled_update = (
            torch.compile(_pure_single_update, backend="tpu", fullgraph=True)
            if compile_update
            else _pure_single_update
        )

    @torch.no_grad()
    def initial_working_params(self) -> tuple[torch.Tensor, ...]:
        device = self.flat_param.device
        flat_cpu = self.flat_param.detach().cpu()
        outputs = []
        for offset, numel, shape, _, target_dtype in self.specs:
            t = flat_cpu[offset : offset + numel].reshape(shape).to(dtype=target_dtype).clone()
            outputs.append(t.to(device=device))
        return tuple(outputs)

    @torch.no_grad()
    def pack_gradients(self, gradients) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for g, pad_len in zip(gradients, self.pad_lengths, strict=True):
            pieces.append(g.detach().float().reshape(-1))
            if pad_len:
                pieces.append(self.zero_pad[:pad_len])
        return torch.cat(pieces, dim=0).contiguous()

    @torch.no_grad()
    def _update_and_emit(self, flat_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
        out = self._compiled_update(
            self.step_tensor, self.flat_param, self.flat_m, self.flat_v, flat_grad
        )
        self.step_tensor = out[0]
        self.flat_param = out[1]
        self.flat_m = out[2]
        self.flat_v = out[3]
        working = out[4:]
        if any(t._base is not None for t in working):
            working = tuple(t.clone() if t._base is not None else t for t in working)
        return working

    @torch.no_grad()
    def step_flat(self, flat_grad: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.reduce_grads:
            reduced = flat_grad.clone()
            torch.distributed.all_reduce(reduced)
            flat_grad = reduced
        return self._update_and_emit(flat_grad)

    @torch.no_grad()
    def copy_master_to_model(self) -> None:
        flat_cpu = self.flat_param.detach().cpu()
        for p, (offset, numel, shape, _, _) in zip(self.params, self.specs, strict=True):
            master = flat_cpu[offset : offset + numel].reshape(shape).to(dtype=p.dtype).clone()
            p.copy_(master)

    @property
    def master_params(self) -> list[torch.Tensor]:
        return [
            self.flat_param[offset : offset + numel].reshape(shape).clone()
            for offset, numel, shape, _, _ in self.specs
        ]

    @property
    def m(self) -> list[torch.Tensor]:
        return [
            self.flat_m[offset : offset + numel].reshape(shape).clone()
            for offset, numel, shape, _, _ in self.specs
        ]

    @property
    def v(self) -> list[torch.Tensor]:
        return [
            self.flat_v[offset : offset + numel].reshape(shape).clone()
            for offset, numel, shape, _, _ in self.specs
        ]


def _model_architecture_signature(loss_model: nn.Module) -> tuple:
    param_sig = tuple((name, tuple(p.shape), p.dtype) for name, p in loss_model.named_parameters())
    if loss_model.workload == "synthetic_transcriptomics_mlp":
        return (loss_model.workload, param_sig)
    vit = loss_model.model
    blocks_sig = tuple(
        (float(b.norm1.eps), int(b.attention.num_heads), float(b.norm2.eps)) for b in vit.blocks
    )
    return (
        loss_model.workload,
        param_sig,
        tuple(vit.patch_embedding.stride),
        tuple(vit.patch_embedding.padding),
        int(vit.norm.normalized_shape[0]),
        float(vit.norm.eps),
        blocks_sig,
    )


def _build_single_microbatch_loss(loss_model: nn.Module, precision: str):
    """Build a pure stateless ATen loss function matching LossModel under autocast."""
    compute_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
    actual_names = [name for name, _ in loss_model.named_parameters()]
    name_to_idx = {name: idx for idx, name in enumerate(actual_names)}

    if loss_model.workload == "synthetic_transcriptomics_mlp":
        expected_mlp_names = (
            "model.layers.0.weight",
            "model.layers.0.bias",
            "model.layers.2.weight",
            "model.layers.2.bias",
        )
        if set(actual_names) != set(expected_mlp_names) or len(actual_names) != len(
            expected_mlp_names
        ):
            raise ValueError(f"Unexpected MLP parameter layout: {actual_names}")
        w0, b0, w2, b2 = (name_to_idx[name] for name in expected_mlp_names)

        def mlp_loss(params, features, target, mask):
            x = features.to(compute_dtype) if features.dtype != compute_dtype else features
            hidden = F.gelu(F.linear(x, params[w0], params[b0]))
            prediction = F.linear(hidden, params[w2], params[b2])
            return masked_mse(prediction, target, mask)

        return mlp_loss, 3

    vit = loss_model.model
    expected_names = [
        "model.class_token",
        "model.position_embedding",
        "model.patch_embedding.weight",
        "model.patch_embedding.bias",
    ]
    block_specs = []
    for idx, block in enumerate(vit.blocks):
        attn = block.attention
        if (
            attn.dropout
            or not attn.batch_first
            or attn.bias_k is not None
            or attn.add_zero_attn
        ):
            raise ValueError("Unsupported attention configuration")
        block_param_names = (
            f"model.blocks.{idx}.norm1.weight",
            f"model.blocks.{idx}.norm1.bias",
            f"model.blocks.{idx}.attention.in_proj_weight",
            f"model.blocks.{idx}.attention.in_proj_bias",
            f"model.blocks.{idx}.attention.out_proj.weight",
            f"model.blocks.{idx}.attention.out_proj.bias",
            f"model.blocks.{idx}.norm2.weight",
            f"model.blocks.{idx}.norm2.bias",
            f"model.blocks.{idx}.mlp.0.weight",
            f"model.blocks.{idx}.mlp.0.bias",
            f"model.blocks.{idx}.mlp.2.weight",
            f"model.blocks.{idx}.mlp.2.bias",
        )
        expected_names.extend(block_param_names)
        block_specs.append(
            (
                float(block.norm1.eps),
                int(attn.num_heads),
                float(block.norm2.eps),
                block_param_names,
            )
        )
    tail_names = (
        "model.norm.weight",
        "model.norm.bias",
        "model.head.weight",
        "model.head.bias",
    )
    expected_names.extend(tail_names)
    if set(actual_names) != set(expected_names) or len(actual_names) != len(expected_names):
        missing = sorted(set(expected_names) - set(actual_names))
        extra = sorted(set(actual_names) - set(expected_names))
        raise ValueError(
            f"Unexpected VisionTransformer parameter layout: missing={missing}, extra={extra}"
        )

    cls_idx = name_to_idx["model.class_token"]
    pos_idx = name_to_idx["model.position_embedding"]
    patch_w_idx = name_to_idx["model.patch_embedding.weight"]
    patch_b_idx = name_to_idx["model.patch_embedding.bias"]
    compiled_blocks = tuple(
        (
            norm1_eps,
            num_heads,
            norm2_eps,
            tuple(name_to_idx[name] for name in param_names),
        )
        for norm1_eps, num_heads, norm2_eps, param_names in block_specs
    )
    norm_w_idx, norm_b_idx, head_w_idx, head_b_idx = (
        name_to_idx[name] for name in tail_names
    )

    patch_stride = tuple(vit.patch_embedding.stride)
    patch_padding = tuple(vit.patch_embedding.padding)
    width = int(vit.norm.normalized_shape[0])
    final_norm_eps = float(vit.norm.eps)

    def vit_loss(params, images, labels):
        x = images.to(compute_dtype) if images.dtype != compute_dtype else images
        tokens = F.conv2d(
            x, params[patch_w_idx], params[patch_b_idx], stride=patch_stride, padding=patch_padding
        )
        tokens = tokens.flatten(2).transpose(1, 2)
        b = images.shape[0]
        class_tokens = params[cls_idx].expand(b, -1, -1)
        values = torch.cat((class_tokens, tokens.float()), dim=1) + params[pos_idx]
        length = values.shape[1]

        for norm1_eps, num_heads, norm2_eps, p_idx in compiled_blocks:
            (
                n1_w,
                n1_b,
                in_w,
                in_b,
                out_w,
                out_b,
                n2_w,
                n2_b,
                m0_w,
                m0_b,
                m2_w,
                m2_b,
            ) = p_idx
            head_dim = width // num_heads
            bh = b * num_heads
            scale = head_dim**-0.5

            normed1 = F.layer_norm(values, (width,), params[n1_w], params[n1_b], eps=norm1_eps)
            q_in = normed1.to(compute_dtype) if normed1.dtype != compute_dtype else normed1
            q, k, v = F.linear(q_in, params[in_w], params[in_b]).chunk(3, dim=-1)
            q = (
                q.reshape(b, length, num_heads, head_dim)
                .transpose(1, 2)
                .reshape(bh, length, head_dim)
            )
            k = (
                k.reshape(b, length, num_heads, head_dim)
                .transpose(1, 2)
                .reshape(bh, length, head_dim)
            )
            v = (
                v.reshape(b, length, num_heads, head_dim)
                .transpose(1, 2)
                .reshape(bh, length, head_dim)
            )
            attn = torch.bmm(q, k.transpose(1, 2)) * scale
            attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(compute_dtype)
            attn_out = torch.bmm(attn, v).reshape(b, num_heads, length, head_dim)
            attn_out = attn_out.transpose(1, 2).reshape(b, length, width)
            attended = F.linear(attn_out, params[out_w], params[out_b])
            values = values + attended.float()

            normed2 = F.layer_norm(values, (width,), params[n2_w], params[n2_b], eps=norm2_eps)
            mlp_in = normed2.to(compute_dtype) if normed2.dtype != compute_dtype else normed2
            mlp_hidden = F.gelu(F.linear(mlp_in, params[m0_w], params[m0_b]))
            mlp_out = F.linear(mlp_hidden, params[m2_w], params[m2_b])
            values = values + mlp_out.float()

        cls_norm = F.layer_norm(
            values[:, 0], (width,), params[norm_w_idx], params[norm_b_idx], eps=final_norm_eps
        )
        head_in = cls_norm.to(compute_dtype) if cls_norm.dtype != compute_dtype else cls_norm
        logits = F.linear(head_in, params[head_w_idx], params[head_b_idx])
        return F.cross_entropy(logits.float(), labels)

    return vit_loss, 2


class WholeStepRunner:
    """Cached 1-microbatch Fwd+Bwd+Pack + functional COMPILE_STAGE flat all_reduce + AdamW."""

    def __init__(
        self,
        model: nn.Module,
        accumulation: int,
        lr: float,
        weight_decay: float,
        precision: str,
        example_flat_inputs: tuple[torch.Tensor, ...],
        compile_step: bool = True,
        trace_fx: bool = True,
        world_size: int = 1,
        reduce_grads: bool = False,
    ):
        self.model = model
        self.accumulation = int(accumulation)
        self.accum_scale = 1.0 / self.accumulation
        expected_arity = 3 if model.workload == "synthetic_transcriptomics_mlp" else 2
        if len(example_flat_inputs) != self.accumulation * expected_arity:
            raise ValueError("example_flat_inputs length does not match accumulation * batch_arity")

        self.optimizer = AlignedFlatAdamW(
            model,
            lr=lr,
            weight_decay=weight_decay,
            precision=precision,
            compile_update=False,
            world_size=world_size,
            reduce_grads=reduce_grads,
        )
        self.working_params = self.optimizer.initial_working_params()
        self.num_params = len(self.working_params)
        effective_reduce_grads = self.optimizer.reduce_grads
        mb_input_sig = tuple(
            (tuple(t.shape), t.dtype, str(t.device)) for t in example_flat_inputs[:expected_arity]
        )
        cache_key = (
            _model_architecture_signature(model),
            precision,
            self.accumulation,
            self.optimizer.world_size,
            effective_reduce_grads,
            self.optimizer.lr,
            self.optimizer.weight_decay,
            bool(compile_step),
            bool(trace_fx),
            os.environ.get("TORCH_TPU_SPMD_HANDSHAKE_STAGE", ""),
            self.optimizer.specs,
            mb_input_sig,
        )
        cached_entry = _COMPILED_V6E_CACHE.get(cache_key)
        if cached_entry is not None:
            (
                self._single_mb_grad_and_pack,
                self._reduce_accum_and_update,
                self.batch_arity,
            ) = cached_entry
            self.cache_hit = True
            return

        single_loss_fn, batch_arity = _build_single_microbatch_loss(model, precision)
        self.batch_arity = batch_arity
        num_params = self.num_params
        pad_lengths = self.optimizer.pad_lengths
        accumulation_steps = self.accumulation
        accum_scale = self.accum_scale
        grad_scale = self.optimizer.grad_scale
        lr_val = self.optimizer.lr
        wd_val = self.optimizer.weight_decay
        specs = self.optimizer.specs

        def _single_mb_grad_and_pack(*mb_tensors):
            zero_pad = mb_tensors[0]
            working = mb_tensors[1 : 1 + num_params]
            mb_inputs = mb_tensors[1 + num_params :]
            grads, loss = torch.func.grad_and_value(single_loss_fn, argnums=0)(working, *mb_inputs)
            pieces = []
            for g, pad_len in zip(grads, pad_lengths, strict=True):
                pieces.append(g.float().reshape(-1))
                if pad_len:
                    pieces.append(zero_pad[:pad_len])
            return torch.cat(pieces, dim=0), loss.float()

        def _reduce_accum_and_update(*state_and_micro_outputs):
            step_tensor = state_and_micro_outputs[0]
            flat_param = state_and_micro_outputs[1]
            flat_m = state_and_micro_outputs[2]
            flat_v = state_and_micro_outputs[3]
            micro_grads = state_and_micro_outputs[4 : 4 + accumulation_steps]
            micro_losses = state_and_micro_outputs[4 + accumulation_steps :]
            flat_grad = micro_grads[0]
            step_loss = micro_losses[0]
            for micro in range(1, accumulation_steps):
                flat_grad = flat_grad + micro_grads[micro]
                step_loss = step_loss + micro_losses[micro]
            if accum_scale != 1.0:
                flat_grad = flat_grad * accum_scale
                step_loss = step_loss * accum_scale
            if effective_reduce_grads:
                reduced = flat_grad.clone()
                torch.distributed.all_reduce(reduced)
                scaled_grad = reduced * grad_scale
            elif grad_scale != 1.0:
                scaled_grad = flat_grad * grad_scale
            else:
                scaled_grad = flat_grad
            updated = _functional_adamw_step_and_emit(
                step_tensor,
                flat_param,
                flat_m,
                flat_v,
                scaled_grad,
                lr_val,
                wd_val,
                specs,
            )
            return (*updated[:4], step_loss, *updated[4:])

        if compile_step or trace_fx:
            with torch.enable_grad():
                traced_single_mb = make_fx(_single_mb_grad_and_pack, tracing_mode="fake")(
                    self.optimizer.zero_pad,
                    *self.working_params,
                    *example_flat_inputs[:batch_arity],
                )
            compiled_single_mb = (
                torch.compile(traced_single_mb, backend="tpu", fullgraph=True)
                if compile_step
                else traced_single_mb
            )
        else:
            compiled_single_mb = _single_mb_grad_and_pack

        compiled_reduce_and_update = (
            torch.compile(_reduce_accum_and_update, backend="tpu", fullgraph=True)
            if compile_step
            else _reduce_accum_and_update
        )

        self._single_mb_grad_and_pack = compiled_single_mb
        self._reduce_accum_and_update = compiled_reduce_and_update
        self.cache_hit = False
        _COMPILED_V6E_CACHE[cache_key] = (
            compiled_single_mb,
            compiled_reduce_and_update,
            batch_arity,
        )

    @torch.no_grad()
    def step(self, *flat_inputs: torch.Tensor) -> torch.Tensor:
        arity = self.batch_arity
        zero_pad = self.optimizer.zero_pad
        working = self.working_params
        micro_grads = []
        micro_losses = []
        for micro in range(self.accumulation):
            start = micro * arity
            g_k, l_k = self._single_mb_grad_and_pack(
                zero_pad, *working, *flat_inputs[start : start + arity]
            )
            micro_grads.append(g_k)
            micro_losses.append(l_k)
        out = self._reduce_accum_and_update(
            self.optimizer.step_tensor,
            self.optimizer.flat_param,
            self.optimizer.flat_m,
            self.optimizer.flat_v,
            *micro_grads,
            *micro_losses,
        )
        self.optimizer.step_tensor = out[0]
        self.optimizer.flat_param = out[1]
        self.optimizer.flat_m = out[2]
        self.optimizer.flat_v = out[3]
        step_loss = out[4]
        new_working = out[5:]
        if any(t._base is not None for t in new_working):
            new_working = tuple(t.clone() if t._base is not None else t for t in new_working)
        self.working_params = new_working
        return step_loss

    @torch.no_grad()
    def sync_model_parameters(self) -> None:
        self.optimizer.copy_master_to_model()


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
