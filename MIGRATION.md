# Migrating this project to the new server

**The two servers share `/media/pinas`** (NFS, `151.100.189.54:/volume1/homes`, 35 TB). All the heavy
data already lives there and does **not** move: the 76 task adapters (27 GB), the PI0.5 base model
(6.8 GB), the external experts (4.1 GB), the demos, and the 654 GB dnd dump.

Only `/home/lzhou` is local (ext4) — **1.6 GB to transfer.** Use `rsync`, not wormhole: you will run it
twice (a pre-seed now, a delta after the running experiments land) and rsync only resends what changed.

> The NFS is mounted `noexec`. The venv and the uv cache must stay on local disk.

---

## What you have to do

### On the NEW server — now, in parallel with the running experiments

**You do not need root.** First find out whether the NAS is already there (on shared clusters it
usually is, often under a different mount point):

```bash
ls /media/pinas 2>/dev/null && echo "SAME PATH — nothing further needed"
mount | grep -iE "151\.100\.189\.54|volume1"      # mounted somewhere else?
df -h | grep -i nfs
```

Then one of three cases:

| case | what to do |
|---|---|
| mounted at `/media/pinas` | nothing — paths already match |
| mounted **elsewhere** (e.g. `/mnt/nas`) | `export LIBERO_DATA_ROOT=/mnt/nas`, then run the relocator (below). **No sudo, no code edits by hand.** |
| **not mounted at all** | you cannot mount it without root — ask the admin for the same NFS export, *or* copy the ~38 GB into your home there and point `LIBERO_DATA_ROOT` at it |

Also: use the same home path `/home/lzhou`. If the account differs, rename
`~/.claude/projects/-home-lzhou` to match the new working directory, or Claude will not load the
project memory and continuity is lost. Install `uv`; nothing else yet.

### On THIS server — pre-seed now, repeat after `ext` and `exttgt` land

(They are resumable; do not kill them. Running rsync while they write is safe — the second pass picks
up whatever changed.)

```bash
NEW=lzhou@newserver
rsync -aP --exclude '.venv' \
          --exclude 'results/analysis/decomp_adapters' \
          --exclude 'results/merge_videos' \
          /home/lzhou/libero-eval/  $NEW:/home/lzhou/libero-eval/     # 881 MB
rsync -aP /home/lzhou/libero-local/ $NEW:/home/lzhou/libero-local/    # 641 MB  (LIBERO + assets)
rsync -aP /home/lzhou/.claude/      $NEW:/home/lzhou/.claude/         #  42 MB  (memory, CLAUDE.md)
```

Excluded on purpose:

| excluded | size | why |
|---|---|---|
| `.venv/` | 6.6 GB | must be rebuilt — different GPU arch and CUDA capability |
| `results/analysis/decomp_adapters/` | 13 GB | build scratch, regenerated on demand |
| `results/merge_videos/` | 231 MB | rollout videos, regenerable |

What is irreplaceable is small: **`results/merge/*.jsonl` (11 MB) — every measurement this project has
ever taken.** Also carried: the soup basis (`loo_soup_factors_t*.npz`, 2 × 249 MB, expensive to
recompute), `EXPERIMENTS.md`, the slide decks, and all reports and figures.

### Back on the NEW server — relocate, build, verify

```bash
cd /home/lzhou/libero-eval
export LIBERO_DATA_ROOT=/mnt/nas          # ONLY if the NAS is not at /media/pinas

# 1. repoint every hardcoded path (repo root + NAS root + the 76 paths inside plan.json)
.venv/bin/python sweep/relocate.py         # dry run — lists every change
.venv/bin/python sweep/relocate.py --apply # rewrites in place, keeps .bak, verifies afterwards

# 2. build the environment (never on the noexec NFS mount)
export UV_CACHE_DIR=/home/lzhou/.cache/uv
uv venv && uv pip install -e . && uv pip install h5py

# 3. sanity
.venv/bin/python sweep/paths.py                          # every artifact reachable?
.venv/bin/python -c "import torch;[print(i, torch.cuda.get_device_properties(i)) \
                     for i in range(torch.cuda.device_count())]"
df -h /dev/shm
```

`relocate.py` is idempotent and safe to re-run after any later rsync from the old host. Order matters
only in that `uv venv` must exist before you invoke `.venv/bin/python` — use the system `python3` for
step 1 if you prefer to relocate first.

Then point Claude Code at `/home/lzhou/libero-eval` and have it read **`HANDOFF.md`**.

**Resume the interrupted experiments** by re-running their drivers — they skip everything already
recorded with status `ok` in `results/merge/*.jsonl`:
```bash
tmux new -d -s ext    '.venv/bin/python sweep/external_driver.py 2>&1 | tee -a results/merge_logs/external.log'
tmux new -d -s exttgt '.venv/bin/python sweep/external_target_driver.py 2>&1 | tee -a results/merge_logs/external_target.log'
```
If a job died mid-write, purge its non-`ok` rows first — a `status: "error"` record is the one thing
that can poison a resume.

---

## Machine-specific constants that must be RE-DERIVED, not inherited

These are tuned to the old box (2× Quadro P6000, 24 GB, CUDA capability 6.1, 32 GB `/dev/shm`).
On better GPUs each is likely wrong in a *good* direction — re-measure before running anything big.

| constant | why it was set | what to do on the new box |
|---|---|---|
| `META["n_workers"] = 3` | the build cache is tmpfs and each merged policy is **6.8 GB**; 4 workers × 6.8 + one building exceeded 32 GB → `ENOSPC` | `df -h /dev/shm` → workers ≈ (shm_GB / 6.8) − 1, also bounded by GPU memory (~8 GB per policy server) |
| `gpu_per_worker = [1, 2, 1]` | two usable GPUs at `nvidia-smi` indices 1 and 2 | remap to the new device count |
| `TORCHDYNAMO_DISABLE=1` | P6000 is capability 6.1; Triton requires ≥ 7.0 | **likely unnecessary** — dropping it enables `torch.compile` and should speed up every rollout |
| torch device order ≠ `nvidia-smi` order | a 2 GB display card was `nvidia-smi:0` but `torch:2` | re-check with `get_device_properties` before pinning devices |
| ~0.7 wall-min per rollout | P6000 throughput | re-time one job before trusting any ETA |

**Every ETA in `EXPERIMENTS.md` is derived from the old throughput — do not carry them forward.**
