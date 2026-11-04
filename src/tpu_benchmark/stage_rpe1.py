"""Stage the checked-in RPE1 package without consuming accelerator devices."""

import argparse
import concurrent.futures
import hashlib
import json
import shlex
import subprocess
import tarfile
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--allocation-job", type=int, required=True)
    parser.add_argument("--package", type=Path, required=True, help="Path to the external pinned RPE1 preprocessing package")
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    nodes = {job["node"]: job for job in receipt["jobs"] if job["job_id"] == args.allocation_job}
    if len(nodes) != 4:
        raise ValueError("Use an existing full-slice allocation for all four workers")
    package = args.package.resolve()
    manifest_hash = hashlib.sha256((package / "manifest.json").read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="rpe1-stage-") as temporary:
        archive = Path(temporary) / "package.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(package, arcname="rpe1-package")

        def stage(node, job):
            root = shlex.quote(job["worker_root"])
            source = shlex.quote(job["release_path"] + "/baseline/src")
            command = f"""set -euo pipefail
root={root}
tar -xf - -C "$root"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 UV_CACHE_DIR="$root/cache"
for backend in xla torchtpu; do
    "$root/tools/bin/uv" pip install --python "$root/$backend/bin/python" --no-deps 'h5py==3.16.0' 'scipy==1.18.1'
    "$root/$backend/bin/python" -c 'import h5py, scipy, numpy; print(h5py.__version__, scipy.__version__, numpy.__version__)'
    "$root/tools/bin/uv" pip freeze --python "$root/$backend/bin/python" > "$root/rpe1-$backend-freeze.txt"
done
export PYTHONPATH={source}
"$root/torchtpu/bin/python" -m gpu_pilot.rpe1_data stage --package "$root/rpe1-package" --local "$root/rpe1-prepared"
"""
            with archive.open("rb") as stream:
                result = subprocess.run(
                    [
                        "srun",
                        f"--jobid={args.allocation_job}",
                        "--overlap",
                        "--nodes=1",
                        "--ntasks=1",
                        "--chdir=/tmp",
                        "--nodelist=" + node,
                        "bash",
                        "-ec",
                        command,
                    ],
                    stdin=stream,
                    capture_output=True,
                    text=False,
                    timeout=300,
                )
            log = args.receipt.parent / f"rpe1-stage-{node}.log"
            log.write_bytes(result.stdout + result.stderr)
            if result.returncode:
                raise RuntimeError(f"RPE1 staging failed on {node}; see {log}")
            return {
                "node": node,
                "manifest_sha256": manifest_hash,
                "status": "staged_and_verified",
                "package": job["worker_root"] + "/rpe1-package",
                "prepared": job["worker_root"] + "/rpe1-prepared",
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(stage, node, job) for node, job in nodes.items()]
            results = [future.result() for future in futures]
        (args.receipt.parent / "rpe1-staging.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(results))


if __name__ == "__main__":
    main()
