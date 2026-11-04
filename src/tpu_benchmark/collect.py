"""Collect completed Slurm runs from worker-local disks, including failures.

Run periodically from a lingering systemd user timer. No chat connection needed.
Only receipt-listed jobs/nodes/paths are accessed. Archives are retained verbatim.
"""

import argparse
import fcntl
import json
import os
import subprocess
from pathlib import Path

from report import report_job


def collect(receipt_path):
    receipt = json.loads(receipt_path.read_text())
    destination = receipt_path.parent / "collected"
    destination.mkdir(exist_ok=True)
    # Read completed evidence through our own active allocations when possible;
    # a subsequent long benchmark need not delay collection for hours.
    running = subprocess.run(
        ["squeue", "--noheader", "--user", str(os.getuid()), "--states=RUNNING", "--format=%A|%N"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    active_nodes = {}
    if running.returncode == 0:
        for line in running.stdout.splitlines():
            job_id, nodes = line.strip().split("|", 1)
            expanded = subprocess.run(
                ["scontrol", "show", "hostnames", nodes], capture_output=True, text=True, timeout=15
            )
            if expanded.returncode == 0:
                active_nodes.update({node: job_id for node in expanded.stdout.splitlines()})
    for job in receipt["jobs"]:
        suffix = "-" + job["node"] if job.get("multi_host") else ""
        target = destination / f"job-{job['job_id']}{suffix}.tar.gz"
        if target.exists():
            continue
        state = subprocess.run(
            ["squeue", "--noheader", "--jobs", str(job["job_id"]), "--format=%T"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if state.returncode == 0 and state.stdout.strip():
            continue
        if state.returncode != 0:
            # Completed jobs can disappear from squeue's in-memory history.
            account = subprocess.run(
                [
                    "sacct",
                    "-X",
                    "--noheader",
                    "--parsable2",
                    "--jobs",
                    str(job["job_id"]),
                    "--format=State",
                ],
                capture_output=True,
                text=True,
                timeout=15,
            )
            terminal = {
                "COMPLETED",
                "FAILED",
                "CANCELLED",
                "TIMEOUT",
                "OUT_OF_MEMORY",
                "NODE_FAIL",
                "PREEMPTED",
            }
            states = [
                line.strip().split()[0].rstrip("+|")
                for line in account.stdout.splitlines()
                if line.strip()
            ]
            if account.returncode or not states or not all(value in terminal for value in states):
                continue
        temporary = target.with_suffix(".partial")
        command = [
            "srun",
            "--partition=tpu",
            "--nodes=1",
            "--ntasks=1",
            "--nodelist=" + job["node"],
            "--chdir=/tmp",
            "--time=00:02:00",
            "--immediate=10",
            "tar",
            "--exclude=*_weights.npz",
            "-czf",
            "-",
            "-C",
            job["worker_root"],
            f"results/job-{job['job_id']}"
            if any(name in job.get("kind", "") for name in ("rpe1", "granular-profile", "vit-single"))
            else "results",
            "xla-freeze.txt",
            "torchtpu-freeze.txt",
        ]
        if "rpe1" in job.get("kind", ""):
            command += ["rpe1-xla-freeze.txt", "rpe1-torchtpu-freeze.txt"]
        if job["node"] in active_nodes:
            command[1:1] = ["--jobid=" + active_nodes[job["node"]], "--overlap"]
        with temporary.open("wb") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.PIPE, timeout=100)
        if result.returncode:
            temporary.unlink(missing_ok=True)
            print(result.stderr.decode(errors="replace"), flush=True)
            continue
        # Check the archive before promoting it. Do not extract arbitrary paths.
        subprocess.run(
            ["tar", "-tzf", str(temporary)], stdout=subprocess.DEVNULL, check=True, timeout=15
        )
        temporary.replace(target)
        print(f"Collected job {job['job_id']} to {target}", flush=True)
    for job_id in sorted({job["job_id"] for job in receipt["jobs"]}):
        jobs = [job for job in receipt["jobs"] if job["job_id"] == job_id]
        archives = [
            destination
            / (f"job-{job_id}" + ("-" + job["node"] if job.get("multi_host") else "") + ".tar.gz")
            for job in jobs
        ]
        if all(path.exists() for path in archives):
            report_job(job_id, archives, receipt_path.parent / "reports" / f"job-{job_id}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    receipt = parser.parse_args().receipt
    with receipt.with_suffix(".collector.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("Another collector is active; retry on the next timer tick")
        else:
            collect(receipt)
