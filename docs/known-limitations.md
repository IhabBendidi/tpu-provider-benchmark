# Known limitations

This organization preserves the existing implementation. It does not apply performance changes or reinterpret archived results.

- **Host synchronization:** historical training reads loss every microbatch and performs per-update synchronization and timing reductions. These boundaries can limit TPU fusion and overlap.
- **Single-device timer:** the TPU timing helper creates and reads a device scalar even for one rank. The original GPU helper returns the host timestamp directly in that case.
- **RPE1 input path:** training uses blocking input transfers followed by synchronization. Evaluation returns batch outputs to the CPU for aggregation. Transfer/runtime overhead warrants separate profiling.
- **Optimizer execution:** baseline AdamW remains outside the compiled model path, with `foreach=False` and `fused=False`. A named compiled or functional variant does not establish a measured optimization win.
- **Runtime allocator:** the original logs warned that tcmalloc was not loaded. A compatible allocator configuration requires a controlled hardware comparison.
- **Contiguous variant:** this calls `.contiguous()` on cached inputs, not model parameters. It does nothing when inputs are already contiguous.
- **SDPA variant:** the original attention already reaches SDPA. The variant changes projection/layout; actual TPU kernel selection and performance must be checked.
- **Hardware validation:** CPU correctness tests and launcher dry runs do not validate execution of advanced optimizer, handshake, bucket or combined variants on a TPU.
- **Cluster assumptions:** supplied multi-host Slurm scripts target a four-host v6e slice with four chips per host. Other layouts require adaptation.
- **Historical archives:** some reporting tools reference result paths from the source repository. Dataset and result archives are not distributed here.

Use baseline XProf traces and numerical checks before accepting performance changes. Preserve the original protocol separately from any newly optimized timing or transfer protocol.
