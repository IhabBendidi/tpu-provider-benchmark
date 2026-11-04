"""Bounded whole-shard Zarr readers; local images always use verified O_DIRECT."""

from __future__ import annotations

import fcntl
import json
import math
import mmap
import os
import random
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import boto3  # type: ignore[import-untyped]
import botocore.session  # type: ignore[import-untyped]
import numpy as np
import torch
import zarr
from botocore.config import Config  # type: ignore[import-untyped]
from zarr.core.buffer import default_buffer_prototype
from zarr.storage import MemoryStore

BUCKET = "rivercell-gpu-pilot-tokyo-921972553032"
REGION = "ap-northeast-1"
PREFIX = "datasets/synthetic-v1/imaging-zarr-20261021T114355Z-32-ba0235d6bb59/"
MANIFEST_SHA = "2dab1907b70d4f8e0622f5ffb850cff227bc3472e99d37e5e090b2624ad09085"
SCRATCH = Path("/scratch/ec2-user/gpu-pilot/imaging-zarr-from-s3.zarr")
ORIGINAL = Path("/scratch/ec2-user/gpu-pilot/imaging-zarr-140gb-31.zarr")
REQUEST_CAP = 2_000_000
REQUEST_USD = 0.00000037


def emit(event: str, **values: Any) -> None:
    sys.stdout.write(json.dumps({"event": event, "timestamp": time.time(), **values}) + "\n")
    sys.stdout.flush()


def reserve_requests(ledger: Path, count: int) -> dict[str, Any]:
    """Reserve once in the coordinator; never block indefinitely on an NFS lock."""
    if count <= 0:
        raise ValueError("Reservation must be positive")
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+") as handle:
        deadline = time.monotonic() + 10
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Request-budget lock unavailable after 10 seconds")
                time.sleep(0.05)
        try:
            handle.seek(0)
            raw = handle.read()
            state = json.loads(raw) if raw else {"reserved": 0, "cap": REQUEST_CAP}
            if state["cap"] != REQUEST_CAP or state["reserved"] + count > REQUEST_CAP:
                raise RuntimeError("Shared S3 request budget exhausted; stopping")
            state["reserved"] += count
            state["maximum_usd"] = state["reserved"] * REQUEST_USD
            handle.seek(0)
            handle.truncate()
            json.dump(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
            return dict(state)
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class RequestBudget:
    """Workers enforce pre-reserved local quotas, with no cross-rank file locks.

    Staging/probes may reserve small blocks; training receives a fixed quota from
    its single launcher. Unused reservations are never refunded, even on crashes.
    """

    def __init__(self, ledger: Path, local_limit: int | None = None):
        if local_limit is not None and not 0 < local_limit <= REQUEST_CAP:
            raise ValueError("Invalid rank request quota")
        self.ledger = ledger
        self.local_limit = local_limit
        self.remaining = local_limit or 0
        self.attempts = 0
        self.lock = threading.Lock()

    def take(self) -> None:
        with self.lock:
            if not self.remaining:
                if self.local_limit is not None:
                    raise RuntimeError("Rank S3 request quota exhausted; stopping")
                reserve_requests(self.ledger, 64)
                self.remaining = 64
            self.remaining -= 1
            self.attempts += 1


def role_client() -> Any:
    session = botocore.session.get_session()
    session.set_config_variable("config_file", "/dev/null")
    session.set_config_variable("credentials_file", "/dev/null")
    session.set_config_variable("profile", None)
    # Explicitly exclude every credential provider except the node's IAM role.
    resolver = session.get_component("credential_provider")
    resolver.providers = [p for p in resolver.providers if p.METHOD == "iam-role"]
    return boto3.Session(botocore_session=session).client(
        "s3",
        region_name=REGION,
        endpoint_url=f"https://s3.{REGION}.amazonaws.com",
        config=Config(
            max_pool_connections=32,
            connect_timeout=5,
            read_timeout=45,
            retries={"total_max_attempts": 1},
        ),
    )


class S3Reader:
    def __init__(self, budget: RequestBudget):
        self.client = role_client()
        self.budget = budget
        self.retry_count = 0
        self.lock = threading.Lock()

    def get(self, key: str) -> tuple[bytes, dict[str, float]]:
        started = time.perf_counter()
        for attempt in range(3):
            self.budget.take()
            body = None
            try:
                response = self.client.get_object(Bucket=BUCKET, Key=key, ChecksumMode="ENABLED")
                first_byte = time.perf_counter()
                body = response["Body"]
                blob = body.read()
                if len(blob) != response["ContentLength"]:
                    raise OSError("Truncated S3 body")
                return blob, {
                    "read_seconds": time.perf_counter() - started,
                    "first_byte_seconds": first_byte - started,
                    "bytes": len(blob),
                    "requests": attempt + 1,
                    "retries": attempt,
                    "physical_bytes": 0,
                }
            except Exception as error:
                emit(
                    "s3_failure",
                    operation="GetObject",
                    destination=f"s3://{BUCKET}/{key}",
                    attempt=attempt + 1,
                    error=str(error),
                )
                if "AccessDenied" in str(error) or attempt == 2:
                    raise
                with self.lock:
                    self.retry_count += 1
                time.sleep(0.2 * (attempt + 1))
            finally:
                if body is not None:
                    body.close()
        raise AssertionError("unreachable")

    def head_preflight(self, key: str) -> dict[str, Any]:
        self.budget.take()
        try:
            result = self.client.head_object(Bucket=BUCKET, Key=key)
        except Exception as error:
            emit(
                "s3_failure",
                operation="HeadObject",
                destination=f"s3://{BUCKET}/{key}",
                error=str(error),
            )
            raise
        if result.get("StorageClass", "STANDARD") != "STANDARD":
            raise RuntimeError("Non-Standard object: retrieval pricing must be reviewed")
        if result.get("ServerSideEncryption") not in (None, "AES256"):
            raise RuntimeError(
                "KMS or other encryption: additional request pricing must be reviewed"
            )
        return {
            key: result.get(key)
            for key in ("ContentLength", "StorageClass", "ServerSideEncryption")
        }


def thread_read_bytes() -> int:
    return int(
        dict(line.split(":", 1) for line in Path("/proc/thread-self/io").read_text().splitlines())[
            "read_bytes"
        ]
    )


def direct_read(path: Path) -> tuple[memoryview, dict[str, float]]:
    """Aligned direct I/O with mandatory physical-read evidence, no fallback.

    Anonymous MAP_SHARED memory is an aligned transfer buffer, not a file map.
    EOF may return an unaligned short final read; the requested size is aligned.
    """
    before = thread_read_bytes()
    started = time.perf_counter()
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT | os.O_NOFOLLOW)
    try:
        size = os.fstat(fd).st_size
        capacity = (size + 4095) // 4096 * 4096
        buffer = mmap.mmap(-1, capacity)
        try:
            received = os.preadv(fd, [buffer], 0)
            if received != size:
                raise OSError(f"Short direct read: {received}/{size}: {path}")
            physical = thread_read_bytes() - before
            if physical < size:
                raise RuntimeError(f"Physical read verification failed: {physical}/{size}: {path}")
            read_seconds = time.perf_counter() - started
            blob = memoryview(buffer)[:size]
        except BaseException:
            buffer.close()
            raise
    finally:
        os.close(fd)
    return blob, {
        "read_seconds": read_seconds,
        "bytes": size,
        "physical_bytes": physical,
        "requests": 0,
        "retries": 0,
        "first_byte_seconds": 0,
        "buffer_copy_seconds": 0.0,
    }


def decode_shard(metadata: bytes, blob: bytes | memoryview, shard: int) -> np.ndarray:
    """Use the official Zarr codec stack, including its CRC32C validation."""
    meta = json.loads(metadata)
    ndim = len(meta["shape"])
    key = "c/" + "/".join([str(shard), *(["0"] * (ndim - 1))])
    factory = default_buffer_prototype().buffer.from_bytes
    store = MemoryStore({"zarr.json": factory(metadata), key: factory(blob)}, read_only=True)
    array = zarr.open_array(store=store, mode="r")
    width = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
    return np.asarray(array[shard * width : (shard + 1) * width])


def shard_order(count: int, rank: int, world: int, epoch: int, seed: int) -> list[int]:
    # An affine permutation avoids an O(dataset_size) shuffle table. This helper
    # returns just this rank's current epoch for the small pilot; iterator below
    # uses the same permutation without a list even for petabyte-scale datasets.
    return list(iter_shards(count, rank, world, epoch, seed))


def iter_shards(
    count: int, rank: int, world: int, epoch: int, seed: int, shards_per_batch: int = 4
) -> Iterator[int]:
    if shards_per_batch < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError("Invalid shard assignment parameters")
    rng = random.Random(seed + epoch)
    multiplier = rng.randrange(1, count)
    while math.gcd(multiplier, count) != 1:
        multiplier = (multiplier + 1) % count or 1
    offset = rng.randrange(count)
    usable = count // (world * shards_per_batch) * (world * shards_per_batch)
    if usable == 0:
        raise ValueError("Dataset must supply one complete batch of shards per rank")
    for position in range(rank * shards_per_batch, usable, world * shards_per_batch):
        for j in range(shards_per_batch):
            yield ((position + j) * multiplier + offset) % count


class ShardDataset:
    def __init__(self, source: str, root: Path, remote: S3Reader | None, split: str = "train"):
        self.source, self.root, self.remote, self.split = source, root, remote, split
        zarr.config.set({"async.concurrency": 2, "threading.max_workers": 2})
        self.metadata = {}
        for name in ("images", "labels"):
            key = f"{split}/{name}/zarr.json"
            self.metadata[name] = self.read(key, image=False)[0]
        meta = json.loads(self.metadata["images"])
        self.count = meta["shape"][0]
        self.width = meta["chunk_grid"]["configuration"]["chunk_shape"][0]
        if (
            self.width != 256
            or meta["shape"][1:] != [3, 224, 224]
            or meta["data_type"] != "float32"
        ):
            raise ValueError("Expected the immutable pilot dataset: 256-image float32 NCHW shards")
        self.shards = self.count // self.width

    def read(self, key: str, image: bool) -> tuple[Any, dict[str, float]]:
        if self.source == "s3":
            assert self.remote is not None
            return self.remote.get(PREFIX + key)
        if image:
            return direct_read(self.root / key)
        started = time.perf_counter()
        blob = (self.root / key).read_bytes()
        return blob, {
            "read_seconds": time.perf_counter() - started,
            "bytes": len(blob),
            "physical_bytes": 0,
            "requests": 0,
            "retries": 0,
        }

    def load(self, shard: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        blob, stats = self.read(f"{self.split}/images/c/{shard}/0/0/0", image=True)
        stats["image_bytes"] = len(blob)
        started = time.perf_counter()
        images = decode_shard(self.metadata["images"], blob, shard)
        stats["decode_seconds"] = time.perf_counter() - started
        del blob
        labels_blob, label_stats = self.read(f"{self.split}/labels/c/{shard}", image=False)
        started = time.perf_counter()
        labels = decode_shard(self.metadata["labels"], labels_blob, shard)
        stats["decode_seconds"] += time.perf_counter() - started
        for key, value in label_stats.items():
            stats[key] = stats.get(key, 0) + value
        return images, labels, stats


class BatchLoader:
    """Exactly one batch under construction; one future limits CPU prefetch.

    The consumer owns at most its current batch plus this single next batch.
    Persistent threads share neither image caches nor growing epoch buffers.
    """

    def __init__(
        self, dataset: ShardDataset, rank: int, world: int, workers: int, seed: int,
        batch_size: int = 1024,
    ):
        if batch_size < dataset.width or batch_size % dataset.width:
            raise ValueError("Batch size must be a positive multiple of the image shard size")
        self.dataset, self.rank, self.world, self.seed = dataset, rank, world, seed
        self.batch_size = batch_size
        self.shards_per_batch = batch_size // dataset.width
        if dataset.shards < world * self.shards_per_batch:
            raise ValueError("Dataset cannot supply one full batch per rank")
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.producer = ThreadPoolExecutor(max_workers=1)
        self.epoch = 0
        self.order = iter(iter_shards(dataset.shards, rank, world, 0, seed, self.shards_per_batch))
        self.pending: Future[tuple[torch.Tensor, torch.Tensor, dict[str, float]]] | None = None

    def build(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        started = time.perf_counter()
        indices = []
        for _ in range(self.shards_per_batch):
            try:
                indices.append(next(self.order))
            except StopIteration:
                self.epoch += 1
                self.order = iter(
                    iter_shards(
                        self.dataset.shards, self.rank, self.world, self.epoch, self.seed,
                        self.shards_per_batch,
                    )
                )
                indices.append(next(self.order))
        pin_start = time.perf_counter()
        images = torch.empty((self.batch_size, 3, 224, 224), dtype=torch.float32, pin_memory=True)
        labels = torch.empty(self.batch_size, dtype=torch.int64, pin_memory=True)
        stats = {"pin_allocate_seconds": time.perf_counter() - pin_start, "collate_seconds": 0.0}
        futures: list[Any] = [self.pool.submit(self.dataset.load, index) for index in indices]
        for j, future in enumerate(futures):
            image, label, read_stats = future.result()
            copy_start = time.perf_counter()
            start, stop = j * self.dataset.width, (j + 1) * self.dataset.width
            images[start:stop].copy_(torch.from_numpy(image))
            labels[start:stop].copy_(torch.from_numpy(label))
            stats["collate_seconds"] += time.perf_counter() - copy_start
            for key, value in read_stats.items():
                stats[key] = stats.get(key, 0) + value
            futures[j] = None  # Release each decoded shard immediately after collation.
        stats["batch_build_seconds"] = time.perf_counter() - started
        return images, labels, stats

    def next(self) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        started = time.perf_counter()
        if self.pending is None:
            self.pending = self.producer.submit(self.build)
        batch = self.pending.result()
        self.pending = self.producer.submit(self.build)
        batch[2]["queue_wait_seconds"] = time.perf_counter() - started
        return batch

    def close(self) -> None:
        self.producer.shutdown(wait=True, cancel_futures=True)
        self.pending = None
        self.pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--reserve", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    reserved = reserve_requests(arguments.ledger, arguments.reserve)
    arguments.output.write_text(json.dumps({"job_quota": arguments.reserve, **reserved}))
    emit("request_quota_reserved", job_quota=arguments.reserve, **reserved)
