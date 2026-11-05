import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def launcher():
    spec = importlib.util.spec_from_file_location("tpu_launch", ROOT / "src/tpu_benchmark/launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("variant", list(launcher().VARIANTS))
def test_separate_launcher_preserves_backend_optimizer_and_communication(variant):
    expected = launcher().VARIANTS[variant]
    result = subprocess.run(
        [
            str(ROOT / "launchers" / f"{variant}.sh"),
            "--dry-run",
            "--config",
            "config.toml",
            "--initial-state",
            "initial.npz",
            "--output",
            "output",
            "--nproc-per-node",
            "4",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    record = json.loads(result.stdout)
    command, env = record["command"], record["environment"]
    assert "--preflight" in command
    for flag, value in [
        ("--backend", expected[0]),
        ("--communication", expected[1]),
        ("--bucket-mb", str(expected[3])),
        ("--optimization", expected[4]),
    ]:
        assert command[command.index(flag) + 1] == value
    assert env["TORCH_TPU_DEFER_AND_FUSE"] == str(int(expected[2]))
    if expected[0] == "xla":
        assert "--xla-spawn" in command and "torch.distributed.run" not in command
    else:
        assert "torch.distributed.run" in command and "--nproc_per_node=4" in command


@pytest.mark.parametrize("name,backend", [("rpe1-xla", "xla"), ("rpe1-torchtpu", "torchtpu")])
def test_rpe1_launcher_has_independent_seed_execution(name, backend):
    r = subprocess.run(
        [
            str(ROOT / "launchers" / f"{name}.sh"),
            "--dry-run",
            "--output",
            "output",
            "--package",
            "package",
            "--local",
            "prepared",
            "--selection",
            "selection.json",
            "--phase",
            "final",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    data = json.loads(r.stdout)
    assert data["backend"] == backend and "torch.distributed.run" not in data["command"]
    assert data["command"][1].endswith("rpe1_train.py")


def test_ddp_launcher_rejects_single_chip_and_backend_override():
    common = [
        "--dry-run",
        "--config",
        "config.toml",
        "--initial-state",
        "weights.npz",
        "--output",
        "out",
    ]
    for extra in ([], ["--nproc-per-node", "4", "--backend", "cpu-test"]):
        r = subprocess.run(
            [str(ROOT / "launchers/torchtpu-ddp.sh"), *common, *extra],
            capture_output=True,
            text=True,
        )
        assert r.returncode == 2


def test_full_launcher_keeps_full_schedule():
    r = subprocess.run(
        [
            str(ROOT / "launchers/xla.sh"),
            "--dry-run",
            "--config",
            "config.toml",
            "--initial-state",
            "weights.npz",
            "--output",
            "out",
            "--measurement-kind",
            "full",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--preflight" not in json.loads(r.stdout)["command"]


def test_v6e_opt_launcher_sets_2x2_topology_and_ddp_bucket100():
    r = subprocess.run(
        [
            str(ROOT / "launchers/torchtpu-v6e-opt.sh"),
            "--dry-run",
            "--config",
            "configs/vit-4chip.toml",
            "--initial-state",
            "/tmp/vit-4chip-initial.npz",
            "--output",
            "/tmp/vit-torchtpu-smoke",
            "--nproc-per-node",
            "4",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(r.stdout)
    env, cmd = plan["environment"], plan["command"]
    assert plan["variant"] == "torchtpu-v6e-opt"
    assert env["TORCH_TPU_DEFER_AND_FUSE"] == "1"
    assert env["TORCH_TPU_DEFER_AND_FUSE_MAX_OPS"] == "500"
    assert env["TORCH_TPU_SPMD_HANDSHAKE_STAGE"] == "OFF"
    assert env["TPU_PROCESS_BOUNDS"] == "2,2,1"
    assert env["TORCH_TPU_TOPOLOGY"] == "2,2,1"
    assert "localhost:8471,localhost:8472,localhost:8473,localhost:8474" in env[
        "TORCH_TPU_SLICEBUILDER_ADDRESSES"
    ]
    assert "XLA_PERSISTENT_CACHE_PATH" in env
    assert cmd[cmd.index("--bucket-mb") + 1] == "100"


def test_launcher_forwards_xprof_option():
    r = subprocess.run(
        [
            str(ROOT / "launchers/torchtpu-v6e-opt.sh"),
            "--dry-run",
            "--config",
            "configs/vit-4chip.toml",
            "--initial-state",
            "/tmp/vit-4chip-initial.npz",
            "--output",
            "/tmp/vit-torchtpu-smoke",
            "--nproc-per-node",
            "4",
            "--xprof=/tmp/xprof-trace",
            "--xprof-steps",
            "3",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cmd = json.loads(r.stdout)["command"]
    assert "--xprof=/tmp/xprof-trace" in cmd
    assert cmd[cmd.index("--xprof-steps") + 1] == "3"
