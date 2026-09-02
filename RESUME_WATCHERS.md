# State at session restart (updated 2026-09-02 12:10)

New session: read this, re-arm the watchers below. Keep this file current; sessions restart.
Context: `HANDOFF.md` (project), `EXPERIMENTS.md` (log), `MEMORY.md` (rules — odin is the default
compute host; shiva only with explicit user approval, currently granted for the hypernetwork/fleet
expansion line; one eval orchestrator per machine).

## Running right now (tmux, resumable — do NOT relaunch, just watch)

| host | tmux | experiment | completion signal |
|---|---|---|---|
| odin  | `unified` | EXP-32 unified 84-pool screen, 86 targets (4/86 at 12:00, ~2 days) | `results/analysis/unified_pool_report.txt` with 86 targets; `results/merge_logs/unified.log` |
| shiva | `newval`  | EXP-36 gate (c): new adapter 25 rollouts on its task + fleet-t0 control | session gone; records `results/merge/e47_new_bowl_stove.jsonl`, `e47_ctrl_fleet_t0.jsonl` |

Landed and written up (slides + EXPERIMENTS.md): EXP-30 (v1 dropped), EXP-33, EXP-34, EXP-35 (uniform stays).

Watch: odin — poll `grep -cE "^\[f[0-9]+\]|^\[x" results/merge_logs/unified.log` vs 86, errors
`No space left|server_died|Traceback|quota`. shiva — `ssh -o BatchMode=yes shiva 'tmux has-session -t <s>'`
with retry on exit 255. Beware: appended logs (`tee -a`) keep old error lines — count only new ones.

## What to do when each lands
- **EXP-36 gate (c)**: new adapter should clearly beat the control (a fleet adapter never trained on the
  task). The goal expert scores 96% on this task (e43_base). If the new adapter is competitive -> propose
  the 30-task batch to the user (train LONGER than 1,500 steps: delta norm was 34.8 vs fleet 59.9; aim for
  the fleet's ~10k steps or match norm), ~30 x GPU-hours on shiva's 3 A6000s. If it fails -> debug data
  pipeline (prompt string, state layout, action padding) before scaling. `sync.sh pull` afterwards.
- **EXP-32**: deployment table ALL/fleet/external/WEAK/STRONG -> slide replacing the EXP-28 slide (user
  confirmed). Its external-target slice re-tests EXP-29's negative; its mixed-candidate slice re-tests
  EXP-28's n=2 positive at n=76.

## Known traps (all hit this week — do not rediscover)
- 10-rollout numbers carry ~13pp SD; never select targets by the same noisy measure used as baseline
  (EXP-27 retracted). Small-n maxima evaporate at full N (EXP-33 -> EXP-35). Probe overstatement +5pp
  internal, ~+19pp cross-pool.
- Drivers self-clean orphaned builds at startup (`_clear_orphaned_builds`); shiva quota 45GB soft.
- Never edit a bash script while it is executing. Never `pgrep` for the orchestrator from inside a
  command line that itself contains the pattern (false BUSY).
- tmpfs build cache ~6.8GB per policy; `hostcfg.py` derives workers/ports per machine.
