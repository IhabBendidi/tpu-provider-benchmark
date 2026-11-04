# Configurations

| Config | Workload | Chips | Precision | Global batch | Microbatch per chip | Accumulation |
| --- | --- | --- | --- | --- | --- | --- |
| `mlp-1chip.toml` | Masked-expression MLP | 1 | BF16 | 512 | 32 | 16 |
| `mlp-4chip.toml` | Masked-expression MLP | 4 | BF16 | 512 | 32 | 4 |
| `mlp-16chip.toml` | Masked-expression MLP | 16 | BF16 | 512 | 32 | 1 |
| `vit-1chip.toml` | ViT | 1 | BF16 | 512 | 32 | 16 |
| `vit-4chip.toml` | ViT | 4 | BF16 | 512 | 32 | 4 |
| `vit-16chip.toml` | ViT | 16 | BF16 | 512 | 32 | 1 |

`rpe1-selection.json` records the original validation-selected hyperparameters and dataset manifest. The RPE1 runner uses its own CLI arguments for epochs, batch size and output paths.

TOML files retain the original shared schema and provenance fields. Workload settings, seeds, batch sizes and timing values have not been changed during repository organization. The separate small configs under `src/tpu_benchmark/baseline/config/` belong to the frozen snapshot and CPU checks.
