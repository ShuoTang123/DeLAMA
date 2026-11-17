#!/usr/bin/env bash
# Wait for a PID to exit, then run a run_lm.py spec.  Usage: wait_then_run.sh PID "args..." LOGFILE
cd "$(dirname "$0")/../.."
PID=$1; SPEC=$2; LOG=$3
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-6} MKL_NUM_THREADS=${MKL_NUM_THREADS:-6}
echo "==== run_lm ${SPEC} $(date) ====" >> "$LOG"
python3 src/lm_delama/run_lm.py ${SPEC} >> "$LOG" 2>&1
echo "==== DONE $(date) ====" >> "$LOG"
