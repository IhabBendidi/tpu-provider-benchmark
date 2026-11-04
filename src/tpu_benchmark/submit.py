"""Stage an immutable code release and submit a bounded, logout-safe TPU job."""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


def write_release_archive(archive):
    """Map the checkout into the original flat worker release layout."""
    source = Path(__file__).resolve().parent
    repository = source.parents[1]
    entries = [(path, path.name) for path in sorted(source.glob("*.py"))]
    entries += [
        (source / "baseline", "baseline"),
        (repository / "configs", "config"),
        (repository / "requirements", "requirements"),
        (repository / "scripts/setup/bootstrap-worker.sh", "bootstrap-worker.sh"),
    ]
    entries += [(path, path.name) for path in sorted((repository / "scripts/slurm").glob("*.sh"))]
    with tarfile.open(archive, "w:gz") as tar:
        for path, name in entries:
            tar.add(
                path, arcname=name, filter=lambda info: None if "__pycache__" in info.name else info
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", default="tputrial-tpuworkers-0")
    parser.add_argument(
        "--nodes",
        nargs="+",
        help="Worker hostnames for multi-host phases; existing v6e topology requires four hosts",
    )
    parser.add_argument("--partition", default="tpu")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--after-job", type=int, help="Stage inside this allocation; queue after it"
    )
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=(
            "baseline",
            "compiled_adamw",
            "functional_adamw",
            "bucket50",
            "bucket100",
            "collective_handshake",
            "sdpa",
            "contiguous",
            "combined",
        ),
    )
    parser.add_argument("--dependency", help="Explicit Slurm dependency, e.g. afterok:123")
    parser.add_argument(
        "--phase",
        choices=(
            "smoke",
            "provider-smoke",
            "distributed-smoke",
            "multihost-smoke",
            "multihost-pilot",
            "multihost-optimize",
            "historical-smoke",
            "multihost-historical-smoke",
            "historical-full",
            "multihost-historical-full",
            "multihost-ddp-historical-full",
            "vit-capacity",
            "single-granular-profile",
            "multihost-granular-profile",
            "multihost-vit-single",
            "multihost-rpe1-smoke",
            "multihost-rpe1-full-xla",
            "multihost-rpe1-full-torchtpu",
        ),
        default="smoke",
    )
    args = parser.parse_args()
    multi_host = args.phase.startswith("multihost-")
    nodes = (
        (args.nodes or [f"tputrial-tpuworkers-{i}" for i in range(4)])
        if multi_host
        else [args.node]
    )
    if multi_host and len(nodes) != 4:
        parser.error("The current multi-host v6e worker requires exactly four hosts")
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if not args.receipt.exists():
        args.receipt.write_text(json.dumps({"jobs": []}) + "\n")
    allocation = [
        f"--nodes={len(nodes)}",
        "--ntasks-per-node=1",
        "--gres=tpu:v6e:4",
        "--nodelist=" + ",".join(nodes),
    ]
    if multi_host:
        allocation.append("--layout=v6e=4x4")
    worker = f"/tmp/gpu-pilot-tpu-{os.getuid()}"
    with tempfile.TemporaryDirectory(prefix="tpu-release-") as temp:
        archive = Path(temp) / "code.tar.gz"
        write_release_archive(archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        release_archive = args.receipt.parent / "releases" / (digest + ".tar.gz")
        release_archive.parent.mkdir(exist_ok=True)
        shutil.copyfile(archive, release_archive)
        release = f"{worker}/releases/{digest}"
        stage = f"mkdir -p {shlex.quote(release)}; tar -xzf - -C {shlex.quote(release)}"
        # Pipe bytes separately to each host: no shell argument-size ceiling and
        # no shared filesystem assumption. Staging needs no accelerator devices.
        for node in nodes:
            stage_command = [
                "srun",
                "--partition=" + args.partition,
                "--nodes=1",
                "--ntasks=1",
                "--nodelist=" + node,
                "--chdir=/tmp",
                "--time=00:02:00",
                "--immediate=15",
                "bash",
                "-ec",
                stage,
            ]
            if args.after_job:
                stage_command[1:1] = [f"--jobid={args.after_job}", "--overlap"]
            with archive.open("rb") as stream:
                subprocess.run(stage_command, stdin=stream, check=True, timeout=100)
        script = Path(temp) / "job.sbatch"
        launch_prefix = (
            "srun --kill-on-bad-exit=1 --ntasks-per-node=1 --gres=tpu:v6e:4 --layout=v6e=4x4 "
            if multi_host
            else ""
        )
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"trap 'cp {shlex.quote(worker)}/slurm-$SLURM_JOB_ID.log "
            f"{shlex.quote(worker)}/results/slurm-$SLURM_JOB_ID.log' EXIT\n"
            f"{launch_prefix}bash {shlex.quote(release)}/smoke-worker.sh {shlex.quote(worker)} {shlex.quote(release)} {shlex.quote(args.phase)}\n"
        )
        dependency_spec = args.dependency or (
            f"afterany:{args.after_job}" if args.after_job else None
        )
        dependency = [f"--dependency={dependency_spec}"] if dependency_spec else []
        job_id = (
            subprocess.check_output(
                [
                    "sbatch",
                    *dependency,
                    "--parsable",
                    "--export=ALL,TPU_SINGLE_VARIANT=" + args.variant,
                    "--partition=" + args.partition,
                    *allocation,
                    "--job-name=tpu-" + args.phase,
                    "--chdir=/tmp",
                    "--time=00:35:00"
                    if args.phase == "multihost-vit-single"
                    else "--time=06:00:00"
                    if args.phase.endswith("historical-full")
                    or "rpe1-full" in args.phase
                    or "granular-profile" in args.phase
                    else "--time=01:15:00"
                    if args.phase == "vit-capacity"
                    else "--time=01:30:00"
                    if args.phase
                    in ("multihost-pilot", "multihost-optimize", "multihost-rpe1-smoke")
                    else "--time=00:25:00",
                    "--output=" + worker + "/slurm-%j.log",
                    str(script),
                ],
                text=True,
                timeout=20,
            )
            .strip()
            .split(";")[0]
        )
        receipt = json.loads(args.receipt.read_text())
        for node in nodes:
            receipt["jobs"].append(
                {
                    "job_id": int(job_id),
                    "kind": args.phase,
                    "variant": args.variant if args.phase == "multihost-vit-single" else None,
                    "node": node,
                    "multi_host": multi_host,
                    "worker_root": worker,
                    "release_sha256": digest,
                    "release_path": release,
                    "after_job": args.after_job,
                    "dependency": dependency_spec,
                }
            )
        temporary = args.receipt.with_suffix(".partial")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n")
        temporary.replace(args.receipt)
        print(json.dumps(receipt["jobs"][-1]))


if __name__ == "__main__":
    main()
