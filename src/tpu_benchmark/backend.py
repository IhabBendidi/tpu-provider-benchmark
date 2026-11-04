"""Explicit TPU runtimes. Never silently fall back to CPU."""

import importlib.metadata
import os
from contextlib import nullcontext

import torch


class Backend:
    def __init__(self, name: str):
        self.name = name
        self.rank, self.world_size = 0, 1
        if name == "xla":
            os.environ["PJRT_DEVICE"] = "TPU"
            import torch_xla
            import torch_xla.core.xla_model as xm
            import torch_xla.runtime as xr

            self.xla, self.xm = torch_xla, xm
            self.device = torch_xla.device()
            if xr.device_type() != "TPU":
                raise RuntimeError("PyTorch/XLA did not initialize a TPU")
            self.rank, self.world_size = xr.global_ordinal(), xr.world_size()
        elif name == "torchtpu":
            import torch_tpu  # noqa: F401

            self.rank = int(os.environ.get("RANK", "0"))
            self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
            if self.world_size > 1:
                torch.distributed.init_process_group("tpu_dist")
            self.device = torch.device("tpu")
        elif name == "cpu-test":
            self.device = torch.device("cpu")
        else:
            raise ValueError(name)
        # A real allocation and synchronization must succeed before any result is written.
        probe = torch.ones((), device=self.device)
        self.sync()
        if float(probe.cpu()) != 1:
            raise RuntimeError("Device correctness probe failed")

    def sync(self):
        if self.name == "xla":
            self.xla.sync(wait=True)
        elif self.name == "torchtpu":
            torch.accelerator.synchronize()

    def autocast(self, precision):
        if precision == "fp32":
            return nullcontext()
        return torch.autocast(self.device.type, dtype=torch.bfloat16)

    def microstep(self):
        if self.name == "xla":
            # Bound lazy graphs when the global batch needs many microbatches.
            self.xla.sync(wait=False)

    def reduce(self, value, operation="sum"):
        if self.world_size == 1:
            return value
        if self.name == "xla":
            return self.xm.all_reduce(operation, value)
        op = {"sum": torch.distributed.ReduceOp.SUM, "max": torch.distributed.ReduceOp.MAX}[
            operation
        ]
        torch.distributed.all_reduce(value, op=op)
        return value

    def max_seconds(self, seconds):
        value = self.reduce(torch.tensor(seconds, device=self.device, dtype=torch.float32), "max")
        self.sync()
        return float(value.cpu())

    def optimizer_step(self, optimizer, communication="per-parameter"):
        # Both backends reduce FP32 gradients once per optimizer update, after
        # accumulation. Use the same parameter order and one collective per tensor.
        if self.world_size > 1 and communication == "per-parameter":
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.copy_(self.reduce(parameter.grad) / self.world_size)
        optimizer.step()

    def metadata(self):
        packages = {}
        for name in ("torch", "torch-xla", "torch-tpu", "libtpu", "numpy"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        metadata = {
            "backend": self.name,
            "device": str(self.device),
            "rank": self.rank,
            "world_size": self.world_size,
            "packages": packages,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        if self.name == "xla":
            import torch_xla.debug.metrics as metrics

            metadata["xla_metrics"] = metrics.metrics_report()
        metadata["launch_environment"] = {
            key: value
            for key, value in os.environ.items()
            if key.startswith(("TPU_", "TORCH_TPU_", "PJRT_"))
            or key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
        }
        return metadata

    def close(self):
        if self.name == "torchtpu" and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
