# State at session restart (updated 2026-09-04 05:30)

New session: read this, re-arm the watchers below. Keep this file current; sessions restart.
Context: `HANDOFF.md` (project), `EXPERIMENTS.md` (log), `MEMORY.md` (rules — odin is the default
compute host; shiva may be used autonomously when IDLE and nothing is queued there, and anything the
user schedules on shiva takes priority — rule updated 09-04; one eval orchestrator per machine).

## Running right now (tmux, resumable — do NOT relaunch, just watch)

| host | tmux | experiment | completion signal |
|---|---|---|---|
| shiva | `sigs_ext`   | queued behind var50: activation signatures for plan_ext indices 106-171 (66 variants incl. 6 short) -> act_sigs/task{106..171}.npz | `results/merge_logs/sigs_ext.log` ends [sigs_ext] done |
| odin  | (bg CPU)     | Gram over plan_ext (172 adapters): `PLAN=sweep/plan_ext.json TAG=ext feats106.py --stage gram` -> lora_gramext.npz; log feats_ext_gram.log | — |
| odin  | (bg CPU)     | EXP-43c activation-block capacity sweep (`run_exp43c.sh` -> exp43c_report.txt; run43c.log ends EXP43c DONE). 43a/b DONE: within-fleet signal = W; cross-suite signal = A only; raw-weight encoder below random -> retired. Selector for new targets should read ACTIVATIONS. | — |
| odin  | `unified` | EXP-32 unified 84-pool screen, 86 targets. ODIN REBOOTED 09-03 ~18:00 (killed the session; tmpfs wiped); RELAUNCHED 21:20 via chain_unified.sh — driver resumes from records (17 baselines / 7 commits done). ~2h/target, ~6 days | `results/analysis/unified_pool_report.txt` with 86 targets; `results/merge_logs/unified.log` |
| odin  | (bg)         | `sweep/land_all.sh` (log land_all.log, ends LAND ALL DONE): waits for shiva sigs_ext -> copies signatures -> land_exp42.sh (exp42_report.txt) -> exp44_deploy.py --stage plan -> launches shiva tmux e44 (build + commits, log e44.log ends [e44] done) -> then run `exp44_deploy.py --stage report` on odin after rsyncing e55_* records |

Landed and written up (slides + EXPERIMENTS.md): EXP-30 (v1 dropped), EXP-33, EXP-34, EXP-35 (uniform stays).

Watch: odin — poll `grep -cE "^\[f[0-9]+\]|^\[x" results/merge_logs/unified.log` vs 86, errors
`No space left|server_died|Traceback|quota`. shiva — `ssh -o BatchMode=yes shiva 'tmux has-session -t <s>'`
with retry on exit 255. Beware: appended logs (`tee -a`) keep old lines from PREVIOUS queue versions — both error lines AND
completion markers like 'ALL STAGES COMPLETE'. Always scope a check to the current run's section
(awk from the last START marker), or a watcher will report a stale completion (happened 09-09).
After ANY restart: check `uptime` on both hosts — a real reboot kills tmux and wipes /dev/shm; the drivers
are resumable, so relaunch the chain script (odin: `tmux new -d -s unified "bash sweep/chain_unified.sh 2>&1 | tee -a results/merge_logs/unified.log"`).

## What to do when each lands
- **EXP-40 DONE + SLIDES CORRECTED (09-05)**: seeded 36-seed: control 76 = +0.20 (p=0.39); 106 on identical
  holdouts = +0.72 (p=0.004) -> 0.017pp/task; cross-distribution +0.82 (p<1e-4). Deck updated (carried-
  forward line, hypernetwork slide, correction note). Never quote +1.31 again.
- **EXP-40 report rows** (for reference): D control(76);
  A pooled(106), B fleet-holdout, C cross-distribution. Slope prediction +1.31 -> ~1.85pp. Report honestly
  as a slope test. NOTE: rsync of brace-expanded remote globs to shiva fails silently (Broken pipe) — use
  tar-over-ssh; and pkill patterns must use the [x] bracket trick AND the same command must contain no plain occurrence
  of the matched string anywhere (heredocs, filenames, comments) — or the command kills itself (exit 144;
  happened twice). Safest: kill in its own tiny command, patch/relaunch in the next.
- EXP-41 landed and the S16 slide is written (EXP-37+39+41). Remaining: EXP-32 (odin, ~Sep 9) -> replace the
  external-donor/target/mixed slides. EXP-38 DONE (variants are real targets: cross-seed affinity Spearman 0.15 < test-retest 0.41).
- LANDING RECIPE (EXP-42): (1) rsync e54 records + report; `sweep/labels_ext.py` -> pair_labels_ext.json;
  (2) after sigs_ext: tar-copy act_sigs/task{106..171}.npz to odin; `PLAN=sweep/plan_ext.json TAG=ext FTAG=ext feats106.py --stage assemble`
  (and FTAG=exta48 with DIMA=48); baselines: feats106.baselines() covers e51/e52/e53/e54 via h6;
  (3) hypernet.py --labels pair_labels_ext.json --feats hypernet_feats{ext,exta48}.npz --baselines <write from feats106.baselines()>
  --holdout fleet|new --blocks WA (holdout_max=106 keeps variants train-only) vs the 106 numbers.
- **When EXP-42 lands** (then EXP-44 deployment test on shiva: 16 held-out targets = new_loras_short + new_loras_seeds, model top-3 vs random top-3 vs probe top-3, ~500 rollouts): rsync new_loras_var50/ + e54 records; extend plan (make_plan106-style, indices 106+)
  for 10 EXP-38 variants + 50 EXP-42 variants; signatures on shiva (activation_probe --plan); feats +
  labels at 166; run the study with variants as TRAINING-only targets, blocks WA at DIMA=24 AND DIMA=48 (pre-registered
  from 43c), evaluate on the unchanged B (fleet holdout) and C (x-dist) protocols; compare to EXP-40's
  +0.72 / +0.82 and 43c's +0.79 / +1.05. Signatures needed for new_loras_seeds (10), new_loras_var50 (50)
  AND new_loras_short (6, for EXP-44). Report as the manufactured-targets
  slope test. Then propose the next selection step to the user.
- NOTE: newpool.log is appended (tee -a) and holds a stale ModuleNotFoundError from a failed first
  launch — count only lines after the last hostcfg banner when checking for errors.
- **AFTER THE SHIVA QUEUE (user asked to be reminded)**: propose EXP-38 seed augmentation — retrain a few
  tasks from S16 with different seeds as real hypernetwork augmentation (checkpoints were identical).
- **EXP-32 (the superset run) — USER STANDING INSTRUCTION, restated 09-04: when it lands, UPDATE THE
  SLIDE CONTENT.** It is the superset of EXP-26/27/28/29, so its results REPLACE those slides rather
  than being appended (final-report form only, no logs of superseded attempts). Deployment table
  ALL/fleet/external/WEAK/STRONG; its external-target slice supersedes EXP-29's negative and its
  mixed-candidate slice supersedes EXP-28's n=2 positive at full n.
- THE DECK IS RENAMED AS THE MEETING DATE MOVES (07_09 -> 16_09 on 09-12, by the user). NEVER hardcode
  the dated name: resolve it with `ls -t results/content_for_slides_*.md | head -1`, or you will write a
  landing into a stale deck that nobody presents.
- LANDING PROTOCOL (user, 09-13): when a result lands, update EXPERIMENTS.md + the deck + memory yourself; chat gets a BRIEF summary only (3-6 lines).
- DECK STYLE (user, 09-13): bullets / fragments ONLY, never prose paragraphs, grammar optional. Memory: slide-content-style.
- Slide hygiene applies to every landing, not just EXP-32: the current deck,
  final-form only, ONE slide per question, superseded numbers replaced in place.

## EXP-81 MOVED TO ODIN (09-16 22:00): it never needed shiva, only a free GPU pair. tmux `ood`,
## log results/merge_logs/ood_odin.log, monitor b5mno5qao. 160 cells, ~4,000 rollouts.
## SHIVA IS STILL ONLY NEEDED FOR: the EXP-77 remainder (3 adapters that exist nowhere else).
## DO NOT POLL SHIVA. The user will say when it is free. Then:
Free = no foreign GPU processes (Matteo's training, ~46GB on GPU) for two consecutive checks. Then from odin:
`bash sweep/sync_code.sh` (pushes main_dense.py ood_shift, orchestrator passthrough, ood_robust.py, chain), then
on shiva `tmux new -d -s ood "bash sweep/chain_shiva_ood.sh 2>&1 | tee results/merge_logs/shiva_ood.log"`.
Chain = EXP-77 remainder (3 targets, S16 slide n=4 -> 7) then EXP-81 (4 arms x 2 shifts x 20 targets, e90_*).
Watcher stopped on user instruction (09-14). Launch by hand on the user's word: sync first, then the tmux line above.

## (superseded) SHIVA IS HANDED OVER TO COLLEAGUES (user, 09-11) — do not launch there
A colleague's training job is running on shiva (Matteo, pi05_so101_lights). Our jobs were stopped
cleanly, all results pulled to odin, tmpfs and build scratch cleared, quota 35.8 -> 23.4GB. Keep ODIN
fed; do not touch shiva until the user says so. The idle sentinel now watches odin only.
TO RESUME LATER: `bash sweep/sync_code.sh` from odin, then `bash sweep/queue_shiva17.sh` on shiva.
NOTE: EXP-77 is running on ODIN for the 4 targets whose adapters exist here (new_loras_short). The 3
EXP-72 targets live only in shiva's results/analysis/new_loras_weak, so their draws need shiva (or a
file copy from it) to finish -- deliberately not done while a colleague's training occupies the machine.
Everything is idempotent -- drivers skip work that already has a result record, so the 17 completed
EXP-77 probes will not be repeated. The 4 weak short adapters restored on 09-11 are still in place, so
EXP-77's target list will be the full 7.

## DECKS: current = content_for_slides_28_09.md (user, 09-19). 16_09 final; 23_09 superseded (carried into 28_09).
## sync_code.sh pushes the NEWEST deck by mtime -> 28_09.
## LESSON 09-19: EXP-83's final run landed at 10:33 on 09-18 and sat UNREPORTED for 25h because the retrain was
## relaunched without re-arming a Monitor. Every relaunch re-arms a watcher in the SAME command. No exceptions.

## STATUS 2026-09-17 17:00 -- SHIVA IS BACK, heavy work moved there
| job | where | state |
|---|---|---|
| EXP-77 remainder (3 S16-init targets) | shiva, tmux `work`, log results/merge_logs/shiva_work.log | running, stage 1 |
| EXP-81 OOD robustness (160 cells) | shiva, stage 2 of the same chain | queued; 28 cells already done, transferred from odin |
| EXP-80 blind union | DONE 09-16, 86/86, +1.53pp all / +4.80pp weak | - |
| EXP-82 xattn selector | DONE, closed negative | - |
| EXP-78 subset probing | DONE 09-20: subset WORSE than singleton, -5.07pp paired (p=0.018). closed | - |
| EXP-83 faithful DA-MergeLoRA merger | DONE 09-19: -14.89pp paired vs probe (p=0.005), closed | - |
WATCHER: bzc797130 (shiva chain). ODIN IS IDLE -- its /dev/sda2 is ~100% full (3-5GB free) so anything
that builds adapters is fragile there; shiva has 117GB free and 3x49GB GPUs (hostcfg derives 7 workers).
SSH GOTCHA: ssh lands in /home/lzhou, NOT the repo. Always `cd /home/lzhou/libero-eval &&` first, including
inside tmux new-session, or the script path and the tee log path both fail silently.

## (done) ODIN CPU SIDE-JOB: EXP-82 cross-attention selector
`sweep/chain_exp82.sh`: waits for demo_bank.npz + lora_col_tokens.npz (background builders, logs demo_bank.log /
lora_col_tokens.log) -> base-policy pass over the demo bank on GPU 2 (activation_probe --base, ~15 min, fits
beside one eval worker) -> xattn_selector.py, 36 seeds x 2 folds x 3 variants, CPU. Report:
results/analysis/exp82_report.txt. Reading: obs > feat (paired) => per-column merger earns rollouts; else closed.

## ODIN QUEUE (reordered again 09-13 00:40) -- tmux `blind`, log `results/merge_logs/blind.log`, monitor bh9izwrc0
Order: 1. EXP-79 DONE 09-13 (+1.00 all / +6.29 weak; slide 3 filled).  2. EXP-80 blind union (`blind_pool.py`, e89_p_/e89_c_) ->
REPLACES slide 1's EXP-32 numbers; weak targets first; run `blind_pool.py --dry` for progress/cost.
3. EXP-32 completion.  4. EXP-78 resume.  Slide 1 currently carries a PENDING banner for EXP-80.
The `xprod` chain below is dead (killed 00:13, nothing lost).

## (dead) ODIN QUEUE (reordered 09-13 on user request) -- tmux `xprod`, log `results/merge_logs/xprod.log`
1. EXP-79 external donors -> fleet receivers (`extdonor_fleet.py`, records e88_f{t}, 32 commits). WHEN IT
   LANDS: fill the PENDING slide "Do EXTERNAL donors improve LIBERO-90 receivers?" in the current deck --
   replace the two `pending` cells, rewrite the headline from PENDING to the answer, keep the three slides
   separate (user asked for one slide per cell). Paired analysis vs +2.22 (fleet-only) and +1.74 (mixed).
2. EXP-32 completion DONE 09-14 19:05 (86/86; deck updated; f28 top-3 unchanged in effect).
3. EXP-78 DEQUEUED by the user 09-16 (gate file sweep/_exp78_disabled; delete it to re-enable, 53/240 kept).
   After EXP-80: shiva runs EXP-81 only, then EXP-83 (faithful DA-MergeLoRA merger).
Monitor `bz90gaib0` watches step 1; arm new ones for steps 2 and 3 when step 1 reports done.

## (superseded) ODIN QUEUE (09-12)
| tmux | experiment | completion signal |
|---|---|---|
| ~~`s16draw`~~ | **EXP-77 DONE 09-12: +12.33pp (SE 4.76), 4/4, 12 runs** vs published single-draw +11.00pp. Deck + log updated. Extends to n=7 when shiva returns (3 veryweak adapters live only there). | done |
| `sub` | EXP-78 subset probing, chained behind EXP-77 via `sweep/chain_sub.sh` | `results/merge_logs/sub.log` ends `[sub] done` |
Both have persistent Monitors armed. EXP-78 compares probing 8 balanced 3-SUBSETS (at the committed
dose c/3, 10 rollouts each) against the standard singleton probe, paired on the SAME targets and the
SAME candidate menus -- the comparison arm `e76_t{t}_d{k}` already exists, so only the new arm is paid
for. Analysis when it lands: paired difference vs e76, per draw and pooled; the deck's Q-slide on
"why the method commits a set" is where the result goes.

## Standing operating rules (user, 09-10)
- KEEP BOTH HOSTS BUSY. Idle is a failure. A standing IDLE SENTINEL monitor reports the moment either
  host has no tmux job session and no orchestrator; arm one if it is not running. Arm a per-job Monitor
  in the SAME turn as the launch, never "later" (forgetting one cost ~7h of idle on 09-10).
- SYNC CODE BEFORE LAUNCHING ON SHIVA: `bash sweep/sync_code.sh` pushes drivers, the eval scripts, plans
  and the small analysis inputs, then verifies 0 files still differ. A stale driver already killed one
  run (unified_pool_driver.py was odin-only and EXP-39 died on import).
- Stop and ask only when something CONCLUSIVE emerges; otherwise keep queueing guided by results.

## Known traps (all hit this week — do not rediscover)
- LIBERO replays FIXED initial states (initial_states[episode_idx], seed 7): a 10-rollout probe scores
  states 0-9, a 25-rollout commit scores 0-24, and states 0-9 measured 11.5pp EASIER. The guard is thus
  optimistic by construction; prefer matched state sets when comparing probe/baseline/commit (EXP-56/57).
- SPEND ROLLOUTS ON DRAWS, NOT EPISODES: draw variance (13.6pp) dominates evaluation variance, so more
  episodes per commit does not tighten the headline (EXP-73: SE rose) while more draws does (EXP-74).
- CANDIDATE-DRAW VARIANCE ~13.6pp per target (EXP-69). Per-target draws make it independent and it lands
  inside the across-target SE; a SHARED draw makes it a systematic bias that can flip an experiment's
  sign (that is what EXP-29 was). NEVER share a candidate draw across targets.
- RUN-TO-RUN NOISE: identical weights on identical states moved single targets by up to 30pp between
  runs (EXP-63). Aggregate means are stable to ~0.5pp at n=15; per-target numbers are not evidence.
- 10-rollout numbers carry ~13pp SD; never select targets by the same noisy measure used as baseline
  (EXP-27 retracted). Small-n maxima evaporate at full N (EXP-33 -> EXP-35). Probe overstatement +5pp
  internal, ~+19pp cross-pool.
- CHAIN SCRIPTS MUST USE `set -e` **AND `set -o pipefail`** + per-stage OK/FAILED markers. With `cmd | tee`,
  `set -e` alone reads tee's status and a crash prints DONE (happened twice: 09-16 quota, 09-18 KeyError). chain_blind2.sh had neither and turned a
  quota crash into '=== ALL DONE ===' on 09-16. Current chain: sweep/chain_blind3.sh (tmux blind3).
- BEFORE BELIEVING ANY LEARNED-SELECTOR NULL, CHECK THE INPUT VARIES. EXP-83 fed 85 identical query
  vectors to the hypernetwork and got a textbook donor-blind collapse that looked exactly like the
  EXP-30 result we already believe. Assert per-input variance BEFORE training, not after.
- THE FLOW-MATCHING LOSS IS STOCHASTIC: it samples noise + timestep per call, so two forwards on IDENTICAL
  weights differ. torch.manual_seed before EVERY forward when comparing losses, or you will chase a
  wiring bug that does not exist (09-18).
- RESULT FILES ARE JSONL: a multi-scene target writes ONE OBJECT PER LINE, so `json.load` raises
  "Extra data: line 2" on perfectly good records. Always parse line by line. A watcher that treated
  those as failures nearly aborted a healthy EXP-81 (09-17).
- WATCHERS MUST COUNT status=="ok" RECORDS, NOT FILES. 40 EXP-81 cells were written as parse_fail and the
  monitor happily reported progress (09-16). Also: any plan using main_dense.py-only flags
  (--args.ood-shift, --args.state-offset) MUST set eval_script in the plan; the orchestrator defaults to main.py.
- ODIN'S DISK IS THE LIMIT, NOT THE QUOTA: /dev/sda2 (469GB) sits 100% full, ~3-5GB free, most of it not
  ours. Run CHUNK=2 there. `quota` will look healthy while builds still fail -- check `df` too.
- MERGED ADAPTERS ARE ~1.66GB EACH (not 0.55GB). CHUNK=6 peaks near 10GB of builds; run CHUNK=3 when
  quota headroom is under ~10GB. `CHUNK` is an env override on unified_pool_driver.
- NEVER draw per-target candidates from ONE sequential rng: a skipped target shifts every later draw (EXP-32
  completion re-probed 36 pairs on 09-14). Per-target md5-seeded rngs only, and freeze sets once probed.
- Drivers self-clean orphaned builds from main() ONLY (never at import — an import during a chunk's build
  window deleted live files on 09-04). shiva quota 45GB soft. AFTER EXP-32 ENDS: relaunch chain_unified.sh
  once to refill f28's two missing probes (donors 74, 75), then check whether f28's commit top-3 changed.
- Never edit a bash script while it is executing. Never `pgrep` for the orchestrator from inside a
  command line that itself contains the pattern (false BUSY) — this includes `tmux new-session ... "bash -c
  'while pgrep ...; python merge_orchestrator.py'"`: put such loops in a script FILE (bit me on 09-07, EXP-44).
- tmpfs build cache ~6.8GB per policy; `hostcfg.py` derives workers/ports per machine.
- RAM: never hold all adapters in memory on a machine with a live eval run (172 x fp16 = 16GB took odin to
  0 GB free on 09-06). feats106.gram() now streams per module; keep it that way.

## Status 2026-09-20 ~05:20
| machine | tmux | job | state | watcher |
|---|---|---|---|---|
| odin | e80d4 | EXP-80 draw 4, 14 weak fleet receivers, CHUNK=2 | running | by5vkp1k9 (results/e80draw4.log) |
| odin | e85 | exp85_prior.py 36 seeds x 300 ep, CPU (started 05:55) | running | b8qfzo1ve (results/e85.log) |
| shiva | e80d | EXP-80 draws 2-3 (4 done on odin), 14 weak fleet receivers | running | bgdg73t9o (results/e80draws.log) |
Closed today: EXP-84 (item-response probe, negative), EXP-87 (progress/speed probe score, progress worse, speed = free tie-break).
Landing EXP-85: read results/analysis/exp85_prior_full.txt; if WAP > WA paired p<0.05 and P alone > 0, run the
exp45_deploy.py LABEL-SPACE check with the WAP model before any rollouts (EXP-45: W+A shortlist = random).

## Status 2026-09-20 ~17:00 — both machines IDLE
Done today: EXP-84 (closed), EXP-87 (closed), EXP-85 incl. label-space check (closed), EXP-80 draw-averaging
(final: blind +4.29 on 14 weak LIBERO-90; -5.12 vs fleet-only, p=0.165). Fingerprint pass 76/76 (e92).
Open decision (user): EXP-86 learned success critic — the only remaining "beat the probe" route; heavy.
No tmux jobs running on odin or shiva. Deck 28_09 current.
