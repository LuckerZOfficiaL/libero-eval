#!/bin/bash
# Keep odin and shiva in step.
#
#   ./sync.sh status        what differs, in both directions (safe, read-only)
#   ./sync.sh push          odin -> shiva   (code + results)
#   ./sync.sh pull          shiva -> odin   (code + results)
#   ./sync.sh push-code     code only (fast; use while experiments are writing results)
#
# WHY RSYNC AND NOT GIT for everything: `results/` is in .gitignore — it is 881 MB of append-only
# measurement records, which git handles badly. Code still belongs in git for history (see NOTE at the
# bottom); this script is for keeping two working copies identical, which is a different job.
#
# SAFETY: never uses --delete. Result files are per-job and append-only, so a merge of two hosts is
# well-defined UNLESS the same job name was computed on both — which is why the operating rule is ONE
# primary compute host at a time (currently: shiva).
set -euo pipefail
REMOTE=${REMOTE:-shiva}
LOCAL=/home/lzhou/libero-eval
REMOTE_DIR=/home/lzhou/libero-eval
EXCL=(--exclude '.venv' --exclude '__pycache__' --exclude '*.bak'
      --exclude 'results/analysis/decomp_adapters'   # regenerable build scratch (13 GB)
      --exclude 'results/merge_videos'               # regenerable rollout videos
      --exclude '.git')
CODE=(--include 'sweep/***' --include 'scripts/***' --include 'src/***' --include 'examples/***'
      --include 'configs/***' --include '*.md' --include '*.sh' --include '*.toml'
      --exclude '*')

case "${1:-status}" in
  status)
    echo "=== would go odin -> shiva ==="
    rsync -an --info=name "${EXCL[@]}" "$LOCAL/" "$REMOTE:$REMOTE_DIR/" | grep -v '/$' | head -25
    echo "=== would go shiva -> odin ==="
    rsync -an --info=name "${EXCL[@]}" "$REMOTE:$REMOTE_DIR/" "$LOCAL/" | grep -v '/$' | head -25
    echo "=== result counts ==="
    echo "  odin : $(ls "$LOCAL"/results/merge/*.jsonl 2>/dev/null | wc -l) files"
    echo "  shiva: $(ssh "$REMOTE" "ls $REMOTE_DIR/results/merge/*.jsonl 2>/dev/null | wc -l") files"
    ;;
  push)      rsync -a --info=progress2 "${EXCL[@]}" "$LOCAL/" "$REMOTE:$REMOTE_DIR/"; echo "pushed odin -> shiva" ;;
  pull)      rsync -a --info=progress2 "${EXCL[@]}" "$REMOTE:$REMOTE_DIR/" "$LOCAL/"; echo "pulled shiva -> odin" ;;
  push-code) rsync -a --info=name "${CODE[@]}" "$LOCAL/" "$REMOTE:$REMOTE_DIR/"; echo "pushed code only" ;;
  pull-code) rsync -a --info=name "${CODE[@]}" "$REMOTE:$REMOTE_DIR/" "$LOCAL/"; echo "pulled code only" ;;
  *) sed -n '2,9p' "$0"; exit 1 ;;
esac

# NOTE ON GIT. Syncing and versioning are different needs. This script keeps the two hosts identical;
# git gives you history and a remote backup. Both are worth having, but git alone will NOT sync results
# (gitignored). Odin currently carries ~120 uncommitted files — a month of work — on top of a commit
# from the mergeability week. Worth one real commit + push to the `fork` remote, independent of this.
