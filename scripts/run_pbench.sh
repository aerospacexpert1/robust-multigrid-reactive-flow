#!/usr/bin/env bash
# Pressure-solver comparison on recorded systems (thesis Section 5.5): every system written by the
# reacting-flow solver (PDUMP_*) is solved by all ten configurations with five start/test combinations.
#   warm prod / warm strict : recorded guess, production test or strict test
#   p0 prod   / p0 strict   : uniform thermodynamic pressure p0
#   cold strict             : zero guess
# Production test: at most 500 cycles; strict test (1e-10 ||r0||): at most 200 cycles.
# Usage: scripts/run_pbench.sh OUTDIR DUMPDIR [DUMPDIR...]   (threads: THREADS="1 16 56 112")
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); B=$ROOT/bin
OUT=${1:?OUTDIR}; shift; [[ $# -gt 0 ]] || { echo "give at least one dump directory" >&2; exit 2; }
THREADS=${THREADS:-1 16 56 112}
mkdir -p "$OUT"
STD=("RMT:s=16" "MG:r=2:g=1:L=0:c=direct" "MG:r=2:g=2:L=0:c=direct" "MG:r=3:g=1:L=0:c=direct" \
     "MG:r=2:g=1:L=0:c=direct:sm=line" "MG:r=2:g=1:L=0:c=direct:sm=line:pre=1:post=1" \
     "MG:r=2:g=2:L=0:c=direct:sm=line:pre=1:post=1" "MG:r=3:g=1:L=0:c=direct:sm=line")
RMTA=("RMTA:s=4" "RMTA:s=6")
export OMP_DYNAMIC=false OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-cores}
for d in "$@"; do
  for sys in "$d"/psys_*.bin; do
    [[ -f $sys ]] || continue
    for th in $THREADS; do
      export OMP_NUM_THREADS=$th
      for sm in "warm prod" "warm strict" "p0 prod" "p0 strict" "cold strict"; do
        read -r st mo <<< "$sm"; mc=500; [[ $mo == strict ]] && mc=200
        "$B/pbench"      "$sys" $st $mo $mc "$OUT/pbench.csv" "$OUT/pbench_hist.csv" "${STD[@]}"  > /dev/null
        "$B/pbench_rmta" "$sys" $st $mo $mc "$OUT/pbench.csv" "$OUT/pbench_hist.csv" "${RMTA[@]}" > /dev/null
      done
    done
    echo "done $sys"
  done
done
