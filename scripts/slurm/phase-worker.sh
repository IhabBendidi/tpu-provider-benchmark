#!/usr/bin/env bash
set -euo pipefail
root=${1:?}
output=${2:?}
phase=${3:?}
mkdir -p "$root/fixtures"
failed=0
configs=(config/mlp-1chip.toml config/vit-1chip.toml)
if [[ "$phase" == multihost-* ]]; then configs=(config/vit-16chip.toml); fi
for config in "${configs[@]}"; do
    workload=$(basename "$config" .toml)
    initial_state="$root/fixtures/job-$SLURM_JOB_ID-$workload-weights.npz"
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$root/torchtpu/bin/python" prepare_initial_state.py \
        --config "$config" --output "$initial_state" > "$output/$workload-initialization.json"
    cases=(32:1)
    if [[ "$workload" == mlp-* ]]; then cases=(32:16 32:1 128:4); fi
    variants=(xla torchtpu torchtpu-defer)
    if [[ "$phase" == multihost-* ]]; then variants+=(torchtpu-ddp torchtpu-ddp-defer); fi
    for case in "${cases[@]}"; do
        batch=${case%:*}
        accumulation=${case#*:}
        for variant in "${variants[@]}"; do
            backend=${variant%%-*}
            export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TORCH_TPU_DEFER_AND_FUSE=0
            [[ "$backend" != torchtpu ]] || export TORCH_DEVICE_BACKEND_AUTOLOAD=1
            [[ "$variant" != *defer ]] || export TORCH_TPU_DEFER_AND_FUSE=1
            command=("$root/$backend/bin/python" phase_profile.py)
            extra=()
            if [[ "$phase" == multihost-* ]]; then
                if [[ "$backend" == xla ]]; then extra+=(--xla-spawn)
                else
                    command=("$root/torchtpu/bin/torchrun" --nnodes=4 --nproc_per_node=4
                        --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" phase_profile.py)
                fi
            fi
            if [[ "$variant" == *ddp* ]]; then extra+=(--communication ddp); fi
            status=0
            timeout 1200 "${command[@]}" --backend "$backend" --config "$config" \
                --initial-state "$initial_state" --microbatch "$batch" --accumulation "$accumulation" \
                --output "$output/phase-profiles/$workload/b${batch}-a${accumulation}/$variant" \
                --trace "${extra[@]}" > "$output/$workload-b${batch}-a${accumulation}-$variant.log" 2>&1 || status=$?
            printf '%s\n' "$status" > "$output/$workload-b${batch}-a${accumulation}-$variant.exit-code"
            if [[ "$status" != 0 ]]; then failed=1; fi
        done
    done
done
exit "$failed"
