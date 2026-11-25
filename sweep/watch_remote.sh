#!/bin/bash
# Watch a job running in tmux on a REMOTE host, from here.
#
#   sweep/watch_remote.sh <tmux-session> <report-path-on-remote> [log-path-on-remote] [poll-sec]
#
# Emits one line per event on stdout, so it plugs straight into the Monitor tool:
#   - "progress: ..."  when the remote log gains a new [done] marker
#   - the report contents, when the report file appears  -> exits 0
#   - a failure notice, if the remote tmux session disappears without a report -> exits 1
#
# ROBUSTNESS: ssh exits 255 on connection failure. A naive check would read a transient network blip
# as "the job died" and fire a false alarm. This distinguishes the two: connection errors are retried
# (up to MAX_SSH_FAIL consecutive) and only a successful ssh reporting no session counts as death.
set -uo pipefail
REMOTE=${REMOTE:-shiva}
SESSION=${1:?tmux session name}
REPORT=${2:?report path on remote}
LOG=${3:-}
POLL=${4:-300}
MAX_SSH_FAIL=${MAX_SSH_FAIL:-10}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10"

fails=0
last=0
while true; do
  # 1. report present?
  out=$($SSH "$REMOTE" "cat '$REPORT' 2>/dev/null" 2>/dev/null); rc=$?
  if [ $rc -eq 255 ]; then
    fails=$((fails+1))
    [ $fails -ge $MAX_SSH_FAIL ] && { echo "=== lost contact with $REMOTE ($fails consecutive ssh failures) ==="; exit 2; }
    sleep 30; continue
  fi
  fails=0
  if [ -n "$out" ]; then
    echo "=== $SESSION COMPLETE on $REMOTE ==="
    echo "$out"
    exit 0
  fi
  # 2. session still alive?
  if ! $SSH "$REMOTE" "tmux has-session -t '$SESSION' 2>/dev/null"; then
    echo "=== $SESSION on $REMOTE is gone and produced no report ==="
    [ -n "$LOG" ] && $SSH "$REMOTE" "tail -15 '$LOG' 2>/dev/null"
    exit 1
  fi
  # 3. progress / errors from the log
  if [ -n "$LOG" ]; then
    n=$($SSH "$REMOTE" "grep -oE '^\[done\][^ ]* [a-zA-Z0-9_]+' '$LOG' 2>/dev/null | sort -u | wc -l" 2>/dev/null)
    n=${n:-0}
    if [ "$n" -gt "$last" ] 2>/dev/null; then
      last=$n
      echo "progress: $n distinct units done"
    fi
    e=$($SSH "$REMOTE" "grep -cE 'No space left|server_died|Traceback|CUDA out of memory' '$LOG' 2>/dev/null" 2>/dev/null)
    [ "${e:-0}" -gt 0 ] 2>/dev/null && echo "=== ${e} error signature(s) in $LOG on $REMOTE ==="
  fi
  sleep "$POLL"
done
