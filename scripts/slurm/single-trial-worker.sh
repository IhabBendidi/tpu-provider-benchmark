#!/usr/bin/env bash
set -euo pipefail
root=${1:?}
output=${2:?}
variant=${TPU_SINGLE_VARIANT:?}
export TORCH_DEVICE_BACKEND_AUTOLOAD=1 TORCH_TPU_DEFER_AND_FUSE=1
export TORCH_TPU_SPMD_HANDSHAKE_STAGE=OFF
config=config/vit-16chip.toml
initial_state="$root/fixtures/job-$SLURM_JOB_ID-vit-weights.npz"
mkdir -p "$root/fixtures"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$root/torchtpu/bin/python" prepare_initial_state.py \
    --config "$config" --output "$initial_state" > "$output/initialization.json"
bucket=25
optimization=$variant
if [[ "$variant" == bucket50 ]]; then bucket=50; optimization=baseline; fi
if [[ "$variant" == bucket100 ]]; then bucket=100; optimization=baseline; fi
command=("$root/torchtpu/bin/torchrun" --nnodes=4 --nproc_per_node=4
    --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT"
    historical_benchmark.py --backend torchtpu --config "$config" --initial-state "$initial_state"
    --communication ddp --bucket-mb "$bucket" --optimization "$optimization" --single-trial)
# Separate preflight output prevents a failed candidate consuming a full trial.
status=0
timeout 240 "${command[@]}" --preflight --output "$output/preflight/$variant" \
    > "$output/preflight-$variant.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$output/preflight-$variant.exit-code"
if [[ "$status" != 0 ]]; then exit "$status"; fi
status=0
timeout 1650 "${command[@]}" --measurement-kind full --output "$output/single-trial/$variant" \
    > "$output/single-trial-$variant.log" 2>&1 || status=$?
printf '%s\n' "$status" > "$output/single-trial-$variant.exit-code"
exit "$status"
