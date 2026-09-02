#!/usr/bin/env python3
"""Derive per-machine orchestrator settings instead of hardcoding them.

Every driver used to carry constants tuned to odin (2x Quadro P6000 24GB, capability 6.1, 32GB
/dev/shm):

    META["n_workers"] = 3          # 4 x 6.8GB policies + one building > 32GB tmpfs -> ENOSPC
    META["gpu_per_worker"] = [1,2,1]
    TORCHDYNAMO_DISABLE=1          # Triton needs capability >= 7.0

Run unchanged on a bigger host and you silently use a fraction of it; run them on a host with a
smaller tmpfs and you get ENOSPC mid-experiment. Both are decided by measurable facts, so measure:

    POLICY_GB   how much a merged PI0.5 policy occupies in the tmpfs build cache
    tmpfs free  bounds how many can be resident at once (plus one being built)
    GPU memory  bounds how many policy servers fit per device
    capability  decides whether torch.compile/Triton is usable

Usage in a driver:

    from hostcfg import apply
    META = apply(META)          # sets n_workers / gpu_per_worker for THIS machine

`python sweep/hostcfg.py` prints what it derived — run it once on any new host.
"""
import json
import os
import pathlib
import shutil
import subprocess

POLICY_GB = 6.8          # measured size of one merged PI0.5 checkpoint in /dev/shm
SERVER_GB = 8.5          # measured GPU memory of one policy server (~8.2GB observed)
MIN_GPU_GB = 16          # ignore display adapters and anything too small to serve a policy


def gpus():
    """[(nvidia_smi_index, name, total_GB, capability), ...] for devices big enough to serve."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return []
    res = []
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 3:
            continue
        idx, name, mem = int(parts[0]), parts[1], float(parts[2]) / 1024.0
        cap = float(parts[3]) if len(parts) > 3 and parts[3] not in ("", "[N/A]") else 0.0
        if mem >= MIN_GPU_GB:
            res.append((idx, name, mem, cap))
    return res


def shm_free_gb():
    try:
        u = shutil.disk_usage("/dev/shm")
        return u.free / 1e9
    except Exception:
        return 0.0


def derive():
    g = gpus()
    shm = shm_free_gb()
    # tmpfs bound: N resident policies + 1 being built must fit, with headroom.
    # ENOSPC has bitten this project twice, and the failure is mid-experiment rather than at launch,
    # so keep 10% of the tmpfs in reserve rather than sizing to the exact limit.
    by_shm = max(1, int((shm * 0.9) / POLICY_GB) - 1)
    # GPU bound: servers that fit across the usable devices
    by_gpu = sum(max(1, int(m / SERVER_GB)) for _, _, m, _ in g) if g else 1
    n = max(1, min(by_shm, by_gpu))
    # round-robin the usable device indices
    order = [idx for idx, _, _, _ in g] or [0]
    gpw = [order[i % len(order)] for i in range(n)]
    caps = [c for _, _, _, c in g]
    return {
        "n_workers": n,
        "gpu_per_worker": gpw,
        "shm_free_gb": round(shm, 1),
        "gpus": [(i, nm, round(m), c) for i, nm, m, c in g],
        "bound_by": "tmpfs" if by_shm <= by_gpu else "gpu-memory",
        "by_shm": by_shm, "by_gpu": by_gpu,
        # Triton (torch.compile) needs capability >= 7.0; below that it raises GPUTooOldForTriton
        "needs_dynamo_disable": bool(caps) and min(caps) < 7.0,
    }


def apply(meta, verbose=True):
    """Overwrite n_workers / gpu_per_worker in an orchestrator META dict for this machine."""
    d = derive()
    meta = dict(meta)
    n = d["n_workers"]
    meta["n_workers"] = n
    meta["gpu_per_worker"] = d["gpu_per_worker"]
    # port_per_worker must be at least as long as n_workers -- the orchestrator indexes PORTS[worker]
    # and a short list is an IndexError at worker start, not a config warning.
    base_ports = list(meta.get("port_per_worker") or [8000])
    start = min(base_ports)
    meta["port_per_worker"] = [start + i for i in range(n)]
    if d["needs_dynamo_disable"]:
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    if verbose:
        print(f"[hostcfg] {d['n_workers']} workers on gpus {d['gpu_per_worker']} "
              f"(bound by {d['bound_by']}; shm {d['shm_free_gb']}GB, {len(d['gpus'])} usable gpus"
              f"{', TORCHDYNAMO_DISABLE set' if d['needs_dynamo_disable'] else ''})", flush=True)
    return meta


if __name__ == "__main__":
    d = derive()
    print(json.dumps(d, indent=1))
    print(f"\n-> n_workers={d['n_workers']}  gpu_per_worker={d['gpu_per_worker']}  "
          f"(bound by {d['bound_by']}: tmpfs allows {d['by_shm']}, gpus allow {d['by_gpu']})")
    print(f"-> TORCHDYNAMO_DISABLE {'REQUIRED' if d['needs_dynamo_disable'] else 'not needed'}")
