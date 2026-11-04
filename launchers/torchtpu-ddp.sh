#!/usr/bin/env bash
set -euo pipefail
repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "${TPU_LAUNCH_PYTHON:-python3}" "$repo_root/src/tpu_benchmark/launch.py" "$@" --variant torchtpu-ddp
