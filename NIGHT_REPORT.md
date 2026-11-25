# Night reports

## Night of 2026-11-10 → 11-11

**Headline** — EXP-157 (LIBERO-10 composites, zero-shot) is NEGATIVE as pre-registered: ours 2% full success vs router-best 4%
(bar was ≥ 20% and ≥ router + 10pp). Every one-model policy (ours, training-free merge, joint training) does one part of the
instruction and stops (~50% of sub-goals); joint fails identically, so this is a single-skill-data limit, not a merging one.
No composition claim; the one-model argument rests on deployment. 18_11 deck slide 5 filled with table + presenter notes.

**Queue status**
- EXP-157, `sweep/chain66_compose_s6.sh` (tmux c66 on shiva): DONE 02:54, 640 episodes, no relaunches.
  Primary 5 tasks, full success | sub-goals: consfz6 0 | 47, consfz6s1 4 | 52, scr 10 | 54, joint24k 2 | 49,
  router-best 4 | 51, base 0 | 1. Per-goal: stove on but moka never placed (0 / 40 one-model episodes); bowl in drawer, drawer
  never closed; first mug yes, pudding never; cream cheese in, soup never. scr's 10% = 5 / 10 on the two-mugs task only.
- Logged: EXPERIMENTS.md EXP-157 (synced to shiva), deck `results/content_for_slides_18_11.md` (slide "Composite
  instructions" + one limitations bullet), memory.
- Queue empty since 02:54; shiva GPUs idle, `/dev/shm` build dir empty, c66 exited by itself. Nothing new started.
- Earlier this evening (in EXPERIMENTS.md): consfz6s1 s6 84.2 / 77.5; consfz6r64 s6 78.2 / 83.8 (not capacity-limited);
  c65 (rank 64 seed 1) futility-stopped 23:08; EXP-150 DONE; 18_11 deck created.

**Bugs hit and fixed**
- 23:08 stopping c65: `pgrep -f` pattern matched my own ssh shell and the follow-up kill ended that shell; trainer stopped via
  `tmux kill-session` as intended; nothing else affected (c64 had finished, c66 untouched). Invalidates nothing.

**Needs your decision**
- Composition follow-up (proposal, not run): a planner that splits the instruction and switches adapters at detected sub-goal
  completion. Single adapters each do their own part (moka-pot adapter: pot on stove 9 / 10), so it would likely chain, but it
  is a routing method (needs the adapter bank + a completion detector) and argues against one model. Suggest: don't run.
- Composition inside one model would need multi-step training data (e.g. own rollouts of chained sub-instructions); new
  direction, your call.
- Held for you: EXP-153 (rollout budget 10 / 20 per skill), EXP-154 (second seed on 4 orders), second backbone / benchmark.
