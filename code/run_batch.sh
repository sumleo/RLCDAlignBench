#!/usr/bin/env bash
# Pilot (20 samples) then full run for a list of sample files with one battery.
# Usage: bash run_batch.sh <battery> <logname> <sample file>...   (run from jev_detect/)
export PYTHONUTF8=1
BAT="$1"; shift; LOG="results/_${1}.log"; shift
: > "$LOG"
for f in "$@"; do
  leg=$(basename "$f" | sed 's/__.*//')
  p=$(python run_jev.py --leg "$leg" --battery "$BAT" --samples "$f" --limit 20 2>&1 | grep -E "^samples:|Traceback|rror" | head -2 | tr '\n' ' ')
  echo "### $(basename "$f")" >> "$LOG"
  echo "pilot: $p" >> "$LOG"
  if echo "$p" | grep -q "api_failed 0"; then
    python run_jev.py --leg "$leg" --battery "$BAT" --samples "$f" 2>&1 | grep -E "^samples:|track/role|Traceback|rror|^\| [a-z]" | head -4 >> "$LOG"
  else
    echo "SKIPPED full run (pilot failed)" >> "$LOG"
  fi
done
echo "BATCH DONE $(date)" >> "$LOG"
grep -E "^###|^samples:|SKIPPED|rror" "$LOG"
