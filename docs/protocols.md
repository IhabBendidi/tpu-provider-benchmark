# Benchmark protocols

Run each comparison with explicit workload, precision, device count, batch size, initialization, data order, and timing settings. Configuration fields are documented in [configs/README.md](../configs/README.md).

## Cached ViT and masked-expression MLP

Reference configs use BF16, global batch 512, and microbatch 32 per device. Configs cover 1, 4, and 16 chips. Accumulation is global_batch / (microbatch × world_size): 16, 4, and 1 respectively. Synthetic cached-input execution repeats the first loaded microbatch, matching the archived protocol.

`prepare_initial_state.py` freezes tensors in an NPZ file and saves the adjacent `.rng.npy` state. Keep both files and reuse them across backends. The historical runner resets model and optimizer at each phase and preserves host loss reads, synchronization and timing instrumentation. The frozen snapshot has its own SHA-256 manifest.

| Run type | Selection | Schedule | Intended use |
| --- | --- | --- | --- |
| Smoke | Default launcher setting | Five preflight updates | Startup and correctness checks |
| Full | `--measurement-kind full` | TOML warmup, repeated timed windows, fixed-update run | Historical protocol comparisons |
| Single trial | `--measurement-kind full --single-trial` | 30-second warmup and one timed window | Separate optimization study |

Reference full configs specify a 600-second warmup, three 1,200-second measurements, and 5,000 fixed-job updates. Single-trial runs omit the fixed job. Five-update smoke rates do not measure sustained throughput. Keep these result types separate.

The schema retains legacy names such as `gpu_only` and `gpu_count`; in TPU configs they describe cached-input execution and chip count. They do not imply CUDA execution.

## RPE1 convergence

RPE1 uses real changing minibatches, FP32 tensors with highest matmul precision, batch 512, and independent seeds. It does not average gradients across seeds. The runner defaults to 150 epochs. The retained selection JSON records the original validation-selected learning rate, dataset manifest, and an earlier 30-epoch final campaign; pass `--epochs 30` when reproducing that duration.

Final runs require the full staged dataset and a matching selection manifest. Smoke limits are explicit runner options. Evaluation uses `no_grad()` because the pinned XLA stack rejected inference tensors. Data sampling, aggregation and metrics remain aligned with the frozen reference.

## Interpreting results

A measured rate describes the complete implementation and software stack. Matching tensor inputs does not establish equal convergence. Equal endpoint metrics do not establish bitwise equality. Stage-fenced diagnostics change overlap and fusion, so their durations are not additive shares of ordinary runtime.

The reference stacks differ: XLA uses PyTorch 2.9.0 and libtpu 0.0.21; TorchTPU uses PyTorch 2.14.1 and libtpu 0.0.51. Archive environment freezes and source hashes with results. Read [known limitations](known-limitations.md) before treating a rate as an architecture ceiling.

[Historical campaign notes](reference-protocol.md) retain the original repository's cluster paths, job numbers and archived result context. Those notes are historical evidence, not commands for this checkout.
