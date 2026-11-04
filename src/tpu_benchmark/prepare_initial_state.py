"""Freeze CPU initialization once; load identical tensors across PyTorch versions."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from workload import make_model, tensor_hash

from gpu_pilot.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    state = make_model(load_config(args.config)).state_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        np.savez(file, **{name: tensor.numpy() for name, tensor in state.items()})
    # Reproduce the original runner's single-rank RandomSampler, whose seed is
    # drawn after model initialization (which differs between torch versions).
    np.save(args.output.with_suffix(".rng.npy"), torch.get_rng_state().numpy())
    print(
        json.dumps(
            {
                "initial_state_sha256": tensor_hash(state.items()),
                "producer_torch": torch.__version__,
                "path": str(args.output),
            }
        )
    )
