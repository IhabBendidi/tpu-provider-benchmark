"""Exercise the unchanged runner's device and BF16 gates; never time CPU as TPU."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "baseline" / "src"))

import torch  # noqa: E402

from gpu_pilot.config import load_config  # noqa: E402
from gpu_pilot.run import initialize_distributed, resolve_precision  # noqa: E402

context = initialize_distributed()
config = load_config(
    Path(__file__).resolve().parent / "baseline/config/transcriptomics.provider-1gpu.toml"
)
record = {
    "torch_version": torch.__version__,
    "selected_device": str(context.device),
    "cuda_available": torch.cuda.is_available(),
    "uses_tpu": context.device.type == "tpu",
}
try:
    record["precision"] = resolve_precision(config, context)
except RuntimeError as error:
    record["provider_profile_error"] = str(error)
print(json.dumps(record, indent=2))
