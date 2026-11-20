#!/usr/bin/env bash
# MMS study of the thesis (Chapter 4): manufactured solution on the production pressure operator,
# aspect ratios AR = 1, 3, 9, 27 on meshes M1-M6, all ten solver configurations.
# Cold start (zero), strict test ||r||_inf <= 1e-10 ||r0||_inf; at most 1000 cycles (300 for the
# line smoothers). On M6 the residual levels off near 2e-10 (round-off), see thesis Chapter 4.
# Usage: scripts/run_mms_study.sh [OUTDIR] [THREADS...]     (defaults: results/mms, 1 16 56)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); B=$ROOT/bin
OUT=${1:-results/mms}; shift || true; THREADS=${*:-1 16 56}
mkdir -p "$OUT/systems"
POINT=("RMT:s=16" "MG:r=2:g=1:L=0:c=direct" "MG:r=2:g=2:L=0:c=direct" "MG:r=3:g=1:L=0:c=direct")
LINE=("MG:r=2:g=1:L=0:c=direct:sm=line" "MG:r=2:g=1:L=0:c=direct:sm=line:pre=1:post=1" \
      "MG:r=2:g=2:L=0:c=direct:sm=line:pre=1:post=1" "MG:r=3:g=1:L=0:c=direct:sm=line")
RMTA=("RMTA:s=4" "RMTA:s=6")
export OMP_DYNAMIC=false OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-cores}
for AR in 1 3 9 27; do
  for M in "108 36" "216 72" "432 144" "864 288" "1728 576" "3456 1152"; do
    read -r NX NY <<< "$M"
    LY=$(awk -v a=$AR 'BEGIN{printf "%.10g", 0.02*a/3}')        # Lx = 0.02 m, Ly = Lx*AR/3
    TAG=AR${AR}_${NX}x${NY}; SYS=$OUT/systems/mms_$TAG.bin
    "$B/mms_gen" "$NX" "$NY" 0.02 "$LY" "$SYS"
    export PBENCH_MMS_OUT=$OUT/mms_error_$TAG.csv
    for th in $THREADS; do
      export OMP_NUM_THREADS=$th
      "$B/pbench"      "$SYS" cold strict 1000 "$OUT/bench_$TAG.csv" "$OUT/hist_$TAG.csv" "${POINT[@]}" > /dev/null
      "$B/pbench_rmta" "$SYS" cold strict 1000 "$OUT/bench_$TAG.csv" "$OUT/hist_$TAG.csv" "${RMTA[@]}"  > /dev/null
      "$B/pbench"      "$SYS" cold strict 300  "$OUT/bench_$TAG.csv" "$OUT/hist_$TAG.csv" "${LINE[@]}"  > /dev/null
    done
    rm -f "$SYS"; echo "done $TAG"
  done
done
