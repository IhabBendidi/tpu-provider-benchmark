"""Explicit launch plans for each TPU backend and tested optimization stack."""

import argparse
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
VARIANTS = {
    "xla": ("xla", "per-parameter", False, 25, "baseline"),
    "torchtpu": ("torchtpu", "per-parameter", False, 25, "baseline"),
    "torchtpu-defer": ("torchtpu", "per-parameter", True, 25, "baseline"),
    "torchtpu-ddp": ("torchtpu", "ddp", False, 25, "baseline"),
    "torchtpu-ddp-defer": ("torchtpu", "ddp", True, 25, "baseline"),
    "torchtpu-compiled-adamw": ("torchtpu", "ddp", True, 25, "compiled_adamw"),
    "torchtpu-functional-adamw": ("torchtpu", "ddp", True, 25, "functional_adamw"),
    "torchtpu-bucket50": ("torchtpu", "ddp", True, 50, "baseline"),
    "torchtpu-bucket100": ("torchtpu", "ddp", True, 100, "baseline"),
    "torchtpu-collective-handshake": ("torchtpu", "ddp", True, 25, "collective_handshake"),
    "torchtpu-sdpa": ("torchtpu", "ddp", True, 25, "sdpa"),
    "torchtpu-contiguous": ("torchtpu", "ddp", True, 25, "contiguous"),
    "torchtpu-combined": ("torchtpu", "ddp", True, 25, "combined"),
    "torchtpu-v6e-opt": ("torchtpu", "ddp", True, 100, "v6e_opt"),
}


def plan(args, extra):
    backend, communication, fusion, bucket, optimization = VARIANTS[args.variant]
    if args.nnodes < 1 or args.nproc_per_node < 1 or not 0 <= args.node_rank < args.nnodes:
        raise ValueError("Invalid distributed node/process counts")
    if communication == "ddp" and args.nnodes * args.nproc_per_node < 2:
        raise ValueError("DDP variants require at least two TPU processes")
    reserved = {"--backend", "--communication", "--bucket-mb", "--optimization", "--xla-spawn"}
    if any(value.split("=", 1)[0] in reserved for value in extra):
        raise ValueError("Use the matching named launcher instead of overriding its fixed settings")
    executable = args.python or os.environ.get(
        "TPU_XLA_PYTHON" if backend == "xla" else "TPU_TORCHTPU_PYTHON",
        str(ROOT / (".venv-xla" if backend == "xla" else ".venv-torchtpu") / "bin/python"),
    )
    environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONUNBUFFERED": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "1" if backend == "torchtpu" else "0",
        "TORCH_TPU_DEFER_AND_FUSE": "1" if fusion else "0",
        "TORCH_TPU_SPMD_HANDSHAKE_STAGE": "OFF",
    }
    if backend == "torchtpu" and args.nnodes == 1 and args.nproc_per_node == 4:
        environment["TPU_PROCESS_BOUNDS"] = os.environ.get("TPU_PROCESS_BOUNDS", "2,2,1")
        environment["TORCH_TPU_TOPOLOGY"] = os.environ.get("TORCH_TPU_TOPOLOGY", "2,2,1")
        environment["TORCH_TPU_DEVICES_PER_HOST"] = os.environ.get(
            "TORCH_TPU_DEVICES_PER_HOST", "4"
        )
        environment["TORCH_TPU_SLICEBUILDER_ADDRESSES"] = os.environ.get(
            "TORCH_TPU_SLICEBUILDER_ADDRESSES",
            "localhost:8471,localhost:8472,localhost:8473,localhost:8474",
        )
    if optimization == "v6e_opt":
        environment["TORCH_TPU_DEFER_AND_FUSE_MAX_OPS"] = os.environ.get(
            "TORCH_TPU_DEFER_AND_FUSE_MAX_OPS", "500"
        )
        environment["XLA_PERSISTENT_CACHE_PATH"] = os.environ.get(
            "XLA_PERSISTENT_CACHE_PATH", "/tmp/tpu-xla-cache"
        )
        for tcmalloc in (
            "/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4",
            "/lib64/libtcmalloc.so.4",
        ):
            if Path(tcmalloc).is_file() and "LD_PRELOAD" not in os.environ:
                environment["LD_PRELOAD"] = tcmalloc
                break
    if backend == "xla":
        environment["PJRT_DEVICE"] = "TPU"
    if args.task == "rpe1":
        if args.variant not in ("xla", "torchtpu-defer"):
            raise ValueError(
                "RPE1 uses XLA or TorchTPU with defer-and-fuse; use its dedicated launcher"
            )
        if args.nnodes * args.nproc_per_node != 1:
            raise ValueError(
                "RPE1 launchers run one independent seed per visible chip; do not use DDP"
            )
        command = [
            executable,
            str(HERE / "rpe1_train.py"),
            "--backend",
            backend,
            "--output",
            str(args.output),
            *extra,
        ]
    else:
        if args.config is None or args.initial_state is None:
            raise ValueError("Training launchers require --config and --initial-state")
        command = [executable]
        if backend == "torchtpu" and args.nnodes * args.nproc_per_node > 1:
            command += [
                "-m",
                "torch.distributed.run",
                f"--nnodes={args.nnodes}",
                f"--nproc_per_node={args.nproc_per_node}",
                f"--node_rank={args.node_rank}",
                f"--master_addr={args.master_addr}",
                f"--master_port={args.master_port}",
                "--max_restarts=0",
            ]
        command += [
            str(HERE / "historical_benchmark.py"),
            "--backend",
            backend,
            "--config",
            str(args.config),
            "--initial-state",
            str(args.initial_state),
            "--output",
            str(args.output),
            "--measurement-kind",
            args.measurement_kind,
            "--communication",
            communication,
            "--bucket-mb",
            str(bucket),
            "--optimization",
            optimization,
        ]
        if backend == "xla" and args.nnodes * args.nproc_per_node > 1:
            command += ["--xla-spawn"]
        if args.measurement_kind == "smoke" and "--preflight" not in extra:
            command += ["--preflight"]
        command += extra
    return {
        "variant": args.variant,
        "task": args.task,
        "backend": backend,
        "environment": environment,
        "command": command,
        "note": "TPU device visibility, topology and multi-host addresses must be supplied by your allocation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--task", choices=("training", "rpe1"), default="training")
    parser.add_argument(
        "--python", help="Interpreter in the selected backend's isolated environment"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--measurement-kind", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    parser.add_argument("--master-addr", default=os.environ.get("MASTER_ADDR", "127.0.0.1"))
    parser.add_argument(
        "--master-port", type=int, default=int(os.environ.get("MASTER_PORT", "29500"))
    )
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    try:
        launch = plan(args, extra)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(launch, indent=2))
        return
    if not Path(launch["command"][0]).is_file():
        parser.error("Backend interpreter missing; install that environment or pass --python")
    subprocess.run(launch["command"], env={**os.environ, **launch["environment"]}, check=True)


if __name__ == "__main__":
    main()
