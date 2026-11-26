#!/usr/bin/env bash
# Pressure-solver comparison on recorded systems (thesis Section 5.5): every system written by the
# reacting-flow solver (PDUMP_*) is solved with five start/test combinations.
#   warm prod / warm strict : recorded guess, production test or strict test
#   p0 prod   / p0 strict   : uniform thermodynamic pressure p0
#   cold strict             : zero guess
# Production test: at most 500 cycles; strict test (1e-10 ||r0||): at most 200 cycles.
#   SOLVERS=main     the ten solver configurations, default (the thesis used THREADS="1 16 56 112")
#   SOLVERS=further  the further references of Section 5.5.5: classical multigrid with the RMTA rule,
#                    parallel memory operations, the fast direct solver, RMT/RMTA with the line
#                    smoother and the combined remedies, with MG2-V and RMTA (s = 6) in the same run
#                    (the thesis used THREADS="1 16 56" and PBENCH_CT=1024)
#   SOLVERS=all      both
# Usage: [SOLVERS=...] [THREADS="1 16 56 112"] scripts/run_pbench.sh OUTDIR DUMPDIR [DUMPDIR...]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); B=$ROOT/bin
OUT=${1:?OUTDIR}; shift; [[ $# -gt 0 ]] || { echo "give at least one dump directory" >&2; exit 2; }
SOLVERS=${SOLVERS:-main}
case $SOLVERS in main|further|all) ;; *) echo "SOLVERS must be main, further or all" >&2; exit 2;; esac
THREADS=${THREADS:-1 16 56 112}
mkdir -p "$OUT"
STD=("RMT:s=16" "MG:r=2:g=1:L=0:c=direct" "MG:r=2:g=2:L=0:c=direct" "MG:r=3:g=1:L=0:c=direct" \
     "MG:r=2:g=1:L=0:c=direct:sm=line" "MG:r=2:g=1:L=0:c=direct:sm=line:pre=1:post=1" \
     "MG:r=2:g=2:L=0:c=direct:sm=line:pre=1:post=1" "MG:r=3:g=1:L=0:c=direct:sm=line")
RMTA=("RMTA:s=4" "RMTA:s=6")
FURTHER=("MG:r=2:g=1:L=0:c=direct" "MG:r=2:g=1:L=0:c=direct:sc=geo" "MG:r=3:g=1:L=0:c=direct:sc=geo" \
         "MG:r=2:g=1:L=0:c=direct:cl=1" "MG:r=2:g=1:L=0:c=direct:sc=geo:cl=1" "FFT" "RMT:s=4:sm=line" \
         "RMT:s=8:sm=line" "RMT:s=16:sm=line" \
         "MG:r=2:g=1:L=0:c=direct:sc=geo:sm=line" "MG:r=3:g=1:L=0:c=direct:sc=geo:sm=line")
FURTHER_RMTA=("RMTA:s=6" "RMTA:s=2:sm=line" "RMTA:s=4:sm=line")
export PBENCH_CT=${PBENCH_CT:-1024}       # cells per thread for cl=1 (the thesis used 1024 here)
export OMP_DYNAMIC=false OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-cores}
for d in "$@"; do
  for sys in "$d"/psys_*.bin; do
    [[ -f $sys ]] || continue
    for th in $THREADS; do
      export OMP_NUM_THREADS=$th
      for sm in "warm prod" "warm strict" "p0 prod" "p0 strict" "cold strict"; do
        read -r st mo <<< "$sm"; mc=500; [[ $mo == strict ]] && mc=200
        if [[ $SOLVERS != further ]]; then
          "$B/pbench"      "$sys" $st $mo $mc "$OUT/pbench.csv" "$OUT/pbench_hist.csv" "${STD[@]}"  > /dev/null
          "$B/pbench_rmta" "$sys" $st $mo $mc "$OUT/pbench.csv" "$OUT/pbench_hist.csv" "${RMTA[@]}" > /dev/null
        fi
        if [[ $SOLVERS != main ]]; then
          "$B/pbench"      "$sys" $st $mo $mc "$OUT/further.csv" "$OUT/further_hist.csv" "${FURTHER[@]}"      > /dev/null
          "$B/pbench_rmta" "$sys" $st $mo $mc "$OUT/further.csv" "$OUT/further_hist.csv" "${FURTHER_RMTA[@]}" > /dev/null
        fi
      done
    done
    echo "done $sys"
  done
done
