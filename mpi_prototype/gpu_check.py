"""Per-rank CUDA binding and verification for 2-rank MPI tests."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass

from mpi4py import MPI


@dataclass(frozen=True)
class GpuBinding:
    mpi_rank: int
    device_index: int
    name: str
    device_id: str  # GPU UUID or PCI bus id — used to prove distinct physical devices
    total_memory_gb: float
    probe_value: float  # small GPU kernel result — proves compute ran on device


def _device_id(device_index: int, props: object) -> str:
    """Stable per-GPU identifier (UUID preferred)."""
    if hasattr(props, "uuid"):
        return str(props.uuid)
    if hasattr(props, "pci_bus_id"):
        return str(props.pci_bus_id)
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=uuid",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return f"cuda:{device_index}:{getattr(props, 'name', 'unknown')}"


def bind_rank_to_cuda(mpi_rank: int) -> GpuBinding:
    """
    Bind this MPI rank to ``cuda:{mpi_rank}`` and run a one-element GPU kernel.

    Raises if CUDA is unavailable or the device index is out of range.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — run on a GPU node with --require-gpu")

    device_count = torch.cuda.device_count()
    if mpi_rank >= device_count:
        raise RuntimeError(
            f"MPI rank {mpi_rank} expects cuda:{mpi_rank} but only "
            f"{device_count} device(s) visible "
            f"(check --gres=gpu:2 and CUDA_VISIBLE_DEVICES)"
        )

    torch.cuda.set_device(mpi_rank)
    device = torch.device(f"cuda:{mpi_rank}")
    props = torch.cuda.get_device_properties(mpi_rank)

    # Minimal device work: unique per rank so we can spot wrong binding.
    x = torch.full((1024,), float(mpi_rank + 1), dtype=torch.float32, device=device)
    y = x * 2.0 + float(mpi_rank)
    torch.cuda.synchronize(device)
    probe_value = float(y[0].item())

    return GpuBinding(
        mpi_rank=mpi_rank,
        device_index=mpi_rank,
        name=props.name,
        device_id=_device_id(mpi_rank, props),
        total_memory_gb=props.total_memory / (1024**3),
        probe_value=probe_value,
    )


def verify_distinct_gpus(comm: MPI.Intracomm, binding: GpuBinding) -> None:
    """All ranks must be bound to different physical GPUs (unique UUID)."""
    rank = comm.Get_rank()
    gathered = comm.allgather(
        (
            binding.mpi_rank,
            binding.device_index,
            binding.device_id,
            binding.name,
            binding.total_memory_gb,
            binding.probe_value,
        )
    )

    uuids = [item[2] for item in gathered]
    if len(set(uuids)) != len(uuids):
        details = "\n".join(
            f"  rank {r}: cuda:{idx} id={did[:20]}… name={name!r} probe={probe:.1f}"
            for r, idx, did, name, _mem, probe in gathered
        )
        raise RuntimeError(
            f"MPI ranks are not on distinct GPUs ({len(set(uuids))} unique ids "
            f"for {len(uuids)} ranks):\n{details}"
        )

    if rank == 0:
        print("GPU binding OK — one physical device per rank:", flush=True)
        for r, idx, did, name, mem_gb, probe in gathered:
            role = "target" if r == 0 else "draft"
            print(
                f"  rank {r} ({role}): cuda:{idx} | {name} | id={did} | "
                f"mem={mem_gb:.1f}GB | probe={probe:.1f}",
                flush=True,
            )


def log_gpu_binding(binding: GpuBinding, *, role: str) -> None:
    print(
        f"[rank {binding.mpi_rank} {role}] cuda:{binding.device_index} "
        f"{binding.name} id={binding.device_id} mem={binding.total_memory_gb:.1f}GB "
        f"probe={binding.probe_value:.1f}",
        flush=True,
    )
