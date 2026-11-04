#!/usr/bin/env bash
set -euo pipefail
root=${1:?Pass the worker workspace}
cd "${2:-$root/code}"
output="$root/results/job-${SLURM_JOB_ID:?Run inside Slurm}"
mkdir -p "$output"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0
python_home=$("$root/xla/bin/python" -c 'import sys; print(sys.base_prefix)')
export LD_LIBRARY_PATH="$python_home/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$root/xla/bin/python" native_probe.py > "$output/native-probe.json" 2> "$output/native-probe.stderr"
export TPU_VISIBLE_CHIPS=0 TPU_VISIBLE_DEVICES=0
export TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1 TPU_PROCESS_ADDRESSES=localhost:8476 TPU_PROCESS_PORT=8476
export WORLD_SIZE=1 RANK=0 LOCAL_RANK=0
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500
export TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v6e-16 CLOUD_TPU_TASK_ID=0
export TPU_WORKER_ID=0 TPU_WORKER_HOSTNAMES=localhost
export TORCH_TPU_TOPOLOGY=1,1,1 TORCH_TPU_SLICEBUILDER_ADDRESSES=localhost:8471
failed=0
phase=${3:-smoke}
warmup_steps=3
steps=5
repetitions=1
limit=300
validation=(--validation-batches 2)
kind=smoke
if [[ "$phase" == multihost-pilot ]]; then
    warmup_steps=10
    steps=1000
    repetitions=3
    limit=1200
    validation=()
    kind=pilot
fi
if [[ "$phase" == multihost-optimize ]]; then
    warmup_steps=10
    steps=100
    repetitions=3
    limit=600
    kind=pilot
fi
if [[ "$phase" == multihost-* ]]; then
    configs=(config/mlp-16chip.toml config/vit-16chip.toml)
    unset TPU_VISIBLE_CHIPS TPU_VISIBLE_DEVICES TPU_PROCESS_ADDRESSES TPU_PROCESS_PORT CLOUD_TPU_TASK_ID
    export TPU_PROCESS_BOUNDS=4,4,1 TPU_HOST_BOUNDS=2,2,1 WORLD_SIZE=16
    export TPU_WORKER_ID="$SLURM_PROCID"
    hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
    TPU_WORKER_HOSTNAMES=$(printf '%s\n' "$hosts" | paste -sd, -)
    export TPU_WORKER_HOSTNAMES
    MASTER_ADDR=$(printf '%s\n' "$hosts" | head -1)
    export MASTER_ADDR TORCH_TPU_TOPOLOGY=4,4,1
    addresses=()
    for host in $hosts; do
        for port in 8471 8472 8473 8474; do addresses+=("$host:$port"); done
    done
    TORCH_TPU_SLICEBUILDER_ADDRESSES=$(IFS=,; printf '%s' "${addresses[*]}")
    export TORCH_TPU_SLICEBUILDER_ADDRESSES
elif [[ "$phase" == distributed-smoke ]]; then
    configs=(config/mlp-4chip.toml config/vit-4chip.toml)
    unset TPU_VISIBLE_CHIPS TPU_VISIBLE_DEVICES TPU_PROCESS_ADDRESSES TPU_PROCESS_PORT CLOUD_TPU_TASK_ID
    export TPU_PROCESS_BOUNDS=2,2,1 WORLD_SIZE=4
    export TORCH_TPU_TOPOLOGY=2,2,1
    export TORCH_TPU_SLICEBUILDER_ADDRESSES=localhost:8471,localhost:8472,localhost:8473,localhost:8474
elif [[ "$phase" == provider-smoke ]]; then
    configs=(config/mlp-1chip.toml config/vit-1chip.toml)
else
    configs=(baseline/config/benchmark.synthetic.local.toml baseline/config/imaging.synthetic.local.toml)
fi
if [[ "$phase" == *historical* ]]; then
    configs=(config/mlp-1chip.toml)
    if [[ "$phase" == multihost-* ]]; then configs=(config/vit-16chip.toml); fi
    if [[ "$phase" == *-full ]]; then limit=7200; else limit=600; fi
fi
if [[ "$phase" == multihost-vit-single ]]; then
    exec bash single-trial-worker.sh "$root" "$output"
fi
if [[ "$phase" == *granular-profile ]]; then
    exec bash phase-worker.sh "$root" "$output" "$phase"
fi
if [[ "$phase" == multihost-rpe1-* ]]; then
    epochs=150
    train_phase=final
    extra=()
    backends=(xla torchtpu)
    limit=21000
    if [[ "$phase" == multihost-rpe1-smoke ]]; then
        epochs=1
        train_phase=smoke
        # Exercise the final partial training batch and all evaluation shapes.
        extra=()
        limit=1200
    elif [[ "$phase" == *-xla ]]; then backends=(xla)
    elif [[ "$phase" == *-torchtpu ]]; then backends=(torchtpu)
    fi
    for backend in "${backends[@]}"; do
        export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TORCH_TPU_DEFER_AND_FUSE=0
        unset XLA_USE_BF16 XLA_DOWNCAST_BF16
        command=("$root/xla/bin/python" rpe1_train.py --xla-spawn)
        if [[ "$backend" == torchtpu ]]; then
            export TORCH_DEVICE_BACKEND_AUTOLOAD=1 TORCH_TPU_DEFER_AND_FUSE=1
            command=("$root/torchtpu/bin/torchrun" --nnodes=4 --nproc_per_node=4
                --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" rpe1_train.py)
        fi
        status=0
        timeout "$limit" "${command[@]}" --backend "$backend" \
            --package "$root/rpe1-package" --local "$root/rpe1-prepared" \
            --output "$output/rpe1/$backend" --phase "$train_phase" \
            --epochs "$epochs" --batch-size 512 --seed-base 1000 \
            --selection config/rpe1-selection.json "${extra[@]}" \
            > "$output/rpe1-$backend.log" 2>&1 || status=$?
        printf '%s\n' "$status" > "$output/rpe1-$backend.exit-code"
        if [[ "$status" != 0 ]]; then failed=1; fi
    done
    exit "$failed"
fi
if [[ "$phase" == vit-capacity ]]; then
    for variant in xla torchtpu torchtpu-defer; do
        backend=${variant%%-*}
        export TORCH_DEVICE_BACKEND_AUTOLOAD=0 TORCH_TPU_DEFER_AND_FUSE=0
        if [[ "$backend" == torchtpu ]]; then export TORCH_DEVICE_BACKEND_AUTOLOAD=1; fi
        if [[ "$variant" == *defer ]]; then export TORCH_TPU_DEFER_AND_FUSE=1; fi
        for batch in 32 1024; do
            status=0
            timeout 600 "$root/$backend/bin/python" capacity.py \
                --backend "$backend" --batch "$batch" --config config/vit-1chip.toml \
                --output "$output/capacity-$variant-$batch.json" \
                > "$output/capacity-$variant-$batch.log" 2>&1 || status=$?
            printf '%s\n' "$status" > "$output/capacity-$variant-$batch.exit-code"
            # A recorded OOM is a valid fit-test result. Unexpected failures are not.
            if [[ "$status" != 0 && "$status" != 42 ]]; then failed=1; fi
        done
    done
    exit "$failed"
fi
for config in "${configs[@]}"; do
workload=$(basename "$config" .toml)
if [[ "$phase" == *historical-smoke ]]; then
    smoke_config="$output/$workload-smoke.toml"
    sed -e 's/warmup_seconds = 600/warmup_seconds = 2/' \
        -e 's/measurement_seconds = 1200/measurement_seconds = 2/' \
        -e 's/measurement_repetitions = 3/measurement_repetitions = 1/' \
        -e 's/fixed_job_updates = 5000/fixed_job_updates = 3/' \
        "$config" > "$smoke_config"
    config="$smoke_config"
fi
initial_state="$root/fixtures/job-$SLURM_JOB_ID-$workload-weights.npz"
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$root/torchtpu/bin/python" prepare_initial_state.py \
    --config "$config" --output "$initial_state" > "$output/$workload-initialization.json"
variants=(xla torchtpu)
if [[ "$phase" == multihost-optimize ]]; then
    variants=(torchtpu torchtpu-ddp torchtpu-defer torchtpu-ddp-defer)
fi
if [[ "$phase" == *historical* ]]; then
    variants=(xla torchtpu torchtpu-defer)
fi
if [[ "$phase" == multihost-ddp-historical-full ]]; then
    variants=(torchtpu-ddp torchtpu-ddp-defer)
fi
for variant in "${variants[@]}"; do
    backend=${variant%%-*}
    export TORCH_TPU_DEFER_AND_FUSE=0
    if [[ "$variant" == *defer ]]; then export TORCH_TPU_DEFER_AND_FUSE=1; fi
    if [[ "$backend" == torchtpu ]]; then
        export TORCH_DEVICE_BACKEND_AUTOLOAD=1
    else
        export TORCH_DEVICE_BACKEND_AUTOLOAD=0
    fi
    command=("$root/$backend/bin/python" benchmark.py)
    extra=()
    if [[ "$phase" == distributed-smoke || "$phase" == multihost-* ]]; then
        if [[ "$backend" == xla ]]; then
            extra=(--xla-spawn)
        else
            command=("$root/$backend/bin/torchrun" --standalone --nproc_per_node=4 benchmark.py)
            if [[ "$phase" == multihost-* ]]; then
                command=("$root/$backend/bin/torchrun" --nnodes=4 --nproc_per_node=4
                    --node_rank="$SLURM_PROCID" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT" benchmark.py)
            fi
        fi
    fi
    if [[ "$variant" == *ddp* ]]; then extra+=(--communication ddp); fi
    status=0
    if [[ "$phase" == *historical* ]]; then
        # Preserve the archived benchmark's reset/duration/host-read protocol.
        command[${#command[@]}-1]=historical_benchmark.py
        historical_kind=smoke
        if [[ "$phase" == *-full ]]; then historical_kind=full; fi
        timeout "$limit" "${command[@]}" "${extra[@]}" \
            --backend "$backend" --config "$config" --initial-state "$initial_state" \
            --output "$output/$workload/$variant" --measurement-kind "$historical_kind" \
            > "$output/$workload-$variant.log" 2>&1 || status=$?
    else
    timeout "$limit" "${command[@]}" "${extra[@]}" \
        --backend "$backend" --config "$config" \
        --initial-state "$initial_state" \
        --output "$output/$workload/$variant" --warmup-steps "$warmup_steps" --steps "$steps" \
        --repetitions "$repetitions" --measurement-kind "$kind" "${validation[@]}" \
        > "$output/$workload-$variant.log" 2>&1 || status=$?
    fi
    printf '%s\n' "$status" > "$output/$workload-$variant.exit-code"
    if [[ "$status" != 0 ]]; then failed=1; fi
done
done
exit "$failed"
