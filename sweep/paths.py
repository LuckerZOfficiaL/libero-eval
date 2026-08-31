#!/usr/bin/env python3
"""Central path resolution — makes the project portable across machines WITHOUT root.

Two things used to be hardcoded in every driver:
  * the repo root, written literally as "/home/lzhou/libero-eval";
  * the NAS data root, written literally as "/media/pinas/...".

Neither is portable, and re-creating `/media/pinas` on a new host needs sudo. Both are now derived:

  REPO       -- from this file's location, so the repo works wherever it is checked out.
  DATA_ROOT  -- $LIBERO_DATA_ROOT if set, else the first path that actually exists from a small
                candidate list (the old mount point first, so nothing changes on the original box).

On a machine where the NAS is mounted somewhere else, the ONLY setup step is:

    export LIBERO_DATA_ROOT=/wherever/the/nas/is        # e.g. /mnt/nas, /shared/homes, $HOME/nasmnt

`python -m sweep.paths` prints what resolved and checks that every expected artifact is reachable —
run it first on a new host.
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent

_CANDIDATES = [
    os.environ.get("LIBERO_DATA_ROOT"),
    "/media/pinas",                      # original host
    "/mnt/pinas", "/mnt/nas", "/shared/homes",
    str(pathlib.Path.home() / "pinas"),
]


def _resolve_data_root():
    for c in _CANDIDATES:
        if c and pathlib.Path(c).is_dir():
            return pathlib.Path(c)
    # Fall back to the original literal so error messages point somewhere meaningful.
    return pathlib.Path("/media/pinas")


DATA_ROOT = _resolve_data_root()

# Everything the project reads from shared storage, in one place.
LORA_POOL = DATA_ROOT / "mehdi/libero90_lora"                 # the 76 task adapters
NORM_STATS = LORA_POOL / "assets/libero90_norm_stats.json"
BASE_MODEL_DIR = DATA_ROOT / "Chris/models/pi05_base_pytorch"  # PI0.5 base policy
BASE_MODEL = BASE_MODEL_DIR / "model.safetensors"
EXT_EXPERTS = DATA_ROOT / "Chris/pi05_libero_lora_experts"     # 8 external suite experts
DEMOS = DATA_ROOT / "palma/inspire_4suite_regenerated_hdf5/libero_90_no_noops"
DND = DATA_ROOT / "mehdi/dnd_libero90_dataset"                 # LoRA checkpoint dump (not needed)

_EXPECTED = {
    "76-adapter pool": (LORA_POOL, True),
    "norm stats": (NORM_STATS, True),
    "base model": (BASE_MODEL, True),
    "external experts": (EXT_EXPERTS, False),
    "demos (optional)": (DEMOS, False),
    "dnd dump (unused)": (DND, False),
}


def check(verbose=True):
    """Return True if every REQUIRED artifact is reachable."""
    ok = True
    if verbose:
        print(f"REPO      = {REPO}")
        print(f"DATA_ROOT = {DATA_ROOT}"
              f"{'  (from $LIBERO_DATA_ROOT)' if os.environ.get('LIBERO_DATA_ROOT') else ''}")
    for name, (p, required) in _EXPECTED.items():
        good = p.exists()
        ok &= good or not required
        if verbose:
            mark = "ok  " if good else ("MISSING" if required else "absent ")
            print(f"  [{mark}] {name:<20} {p}")
    if verbose and not ok:
        print("\nSet LIBERO_DATA_ROOT to the directory that contains mehdi/ and Chris/, e.g.:")
        print("  export LIBERO_DATA_ROOT=/mnt/nas")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if check() else 1)
