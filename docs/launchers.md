# Launcher reference

All paths below are under `launchers/`. Training launchers share `src/tpu_benchmark/launch.py`, which fixes each variant's backend, communication, fusion, optimizer, and bucket settings.

| Launcher | Communication | Defer/fuse | Additional optimization |
| --- | --- | --- | --- |
| `xla.sh` | XLA per-parameter collectives | XLA execution | Baseline AdamW |
| `torchtpu.sh` | Per-parameter collectives | Off | Baseline AdamW |
| `torchtpu-defer.sh` | Per-parameter collectives | On | Baseline AdamW |
| `torchtpu-ddp.sh` | DDP, 25 MiB bucket | Off | Baseline AdamW |
| `torchtpu-ddp-defer.sh` | DDP, 25 MiB bucket | On | Baseline AdamW |
| `torchtpu-compiled-adamw.sh` | DDP, 25 MiB bucket | On | Compiled AdamW |
| `torchtpu-functional-adamw.sh` | DDP, 25 MiB bucket | On | Functional AdamW |
| `torchtpu-bucket50.sh` | DDP, 50 MiB bucket | On | Baseline AdamW |
| `torchtpu-bucket100.sh` | DDP, 100 MiB bucket | On | Baseline AdamW |
| `torchtpu-collective-handshake.sh` | DDP, 25 MiB bucket | On | Collective handshake |
| `torchtpu-sdpa.sh` | DDP, 25 MiB bucket | On | Scaled dot-product attention |
| `torchtpu-contiguous.sh` | DDP, 25 MiB bucket | On | Contiguous cached inputs |
| `torchtpu-combined.sh` | DDP, 25 MiB bucket | On | Functional AdamW + handshake + SDPA + contiguous cached inputs |
| `rpe1-xla.sh` | Independent seed | XLA execution | RPE1 convergence |
| `rpe1-torchtpu.sh` | Independent seed | On | RPE1 convergence |

The additional optimization variants intentionally build on **DDP + defer/fuse**, matching the original optimization study. They are fixed experimental compositions, not an exhaustive Cartesian product. DDP launchers require at least two processes. SDPA affects ViT attention. The handshake and combined variants capture explicit collectives in the compiled graph and bypass the ordinary DDP wrapper. See [vit_optimizations.py](../src/tpu_benchmark/vit_optimizations.py) for implementations.

The contiguous variant operates on input tensors and is a no-op when they are already contiguous. The baseline attention already reaches SDPA; the explicit variant changes the projection/layout path. Neither name establishes a measured speedup. Compiled/functional optimizers and other advanced variants require TPU correctness and performance validation. See [known limitations](known-limitations.md).
