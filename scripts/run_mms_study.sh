#!/usr/bin/env bash
# MMS study of the thesis (Chapter 4): manufactured solution on the production pressure operator,
# aspect ratios AR = 1, 3, 9, 27 on meshes M1-M6.
# Cold start (zero), strict test ||r||_inf <= 1e-10 ||r0||_inf; at most 1000 cycles (300 for the
# line smoothers of the main study). On M6 the residual levels off near 2e-10 (round-off), see
# thesis Chapter 4.
#   SOLVERS=main     the ten solver configurations (Sections 4.2-4.6), default
#   SOLVERS=further  the further references (Section 4.7): classical multigrid with the RMTA rule,
#                    parallel memory operations, RMT/RMTA with the line smoother, the combined
#                    remedies and the fast direct solver, with MG2-V and RMTA (s = 6) in the same run
#   SOLVERS=all      both
# Usage: [SOLVERS=...] scripts/run_mms_study.sh [OUTDIR] [THREADS...]   (defaults: results/mms, 1 16 56)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); B=$ROOT/bin
OUT=${1:-results/mms}; shift || true; THREADS=${*:-1 16 56}
SOLVERS=${SOLVERS:-main}
case $SOLVERS in main|further|all) ;; *) echo "SOLVERS must be main, further or all" >&2; exit 2;; esac
mkdir -p "$OUT/systems"
POINT=("RMT:s=16" "MG:r=2:g=1:L=0:c=direct" "MG:r=2:g=2:L=0:c=direct" "MG:r=3:g=1:L=0:c=direct")
LINE=("MG:r=2:g=1:L=0:c=direct:sm=line" "MG:r=2:g=1:L=0:c=direct:sm=line:pre=1:post=1" \
      "MG:r=2:g=2:L=0:c=direct:sm=line:pre=1:post=1" "MG:r=3:g=1:L=0:c=direct:sm=line")
RMTA=("RMTA:s=4" "RMTA:s=6")
FURTHER=("MG:r=2:g=1:L=0:c=direct" \
         "MG:r=2:g=1:L=0:c=direct:sc=geo" "MG:r=3:g=1:L=0:c=direct:sc=geo" "MG:r=2:g=2:L=0:c=direct:sc=geo" \
         "MG:r=2:g=1:L=0:c=direct:cl=1" "MG:r=2:g=1:L=0:c=direct:sc=geo:cl=1" "FFT" \
         "RMT:s=2:sm=line" "RMT:s=4:sm=line" "RMT:s=8:sm=line" "RMT:s=16:sm=line" \
         "MG:r=2:g=1:L=0:c=direct:sc=geo:sm=line" "MG:r=3:g=1:L=0:c=direct:sc=geo:sm=line")
FURTHER_RMTA=("RMTA:s=6" "RMTA:s=2:sm=line" "RMTA:s=4:sm=line")
export PBENCH_CT=${PBENCH_CT:-4096}       # cells per thread for cl=1 (the thesis used 4096 here)
export OMP_DYNAMIC=false OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-cores}
for AR in 1 3 9 27; do
  for M in "108 36" "216 72" "432 144" "864 288" "1728 576" "3456 1152"; do
    read -r NX NY <<< "$M"
    LY=$(awk -v a=$AR 'BEGIN{printf "%.10g", 0.02*a/3}')        # Lx = 0.02 m, Ly = Lx*AR/3
    TAG=AR${AR}_${NX}x${NY}; SYS=$OUT/systems/mms_$TAG.bin
    "$B/mms_gen" "$NX" "$NY" 0.02 "$LY" "$SYS"
    for th in $THREADS; do
      export OMP_NUM_THREADS=$th
      if [[ $SOLVERS != further ]]; then
        export PBENCH_MMS_OUT=$OUT/mms_error_$TAG.csv
        "$B/pbench"      "$SYS" cold strict 1000 "$OUT/bench_$TAG.csv" "$OUT/hist_$TAG.csv" "${POINT[@]}" > /dev/null
        "$B/pbench_rmta" "$SYS" cold strict 1000 "$OUT/bench_$TAG.csv" "$OUT/hist_$TAG.csv" "${RMTA[@]}"  > /dev/null
        "$B/pbench"      "$SYS" cold strict 300  "$OUT/bench_$TAG.csv" "$OUT/hist_$TAG.csv" "${LINE[@]}"  > /dev/null
      fi
      if [[ $SOLVERS != main ]]; then
        export PBENCH_MMS_OUT=$OUT/further_mms_error_$TAG.csv
        "$B/pbench"      "$SYS" cold strict 1000 "$OUT/further_bench_$TAG.csv" "$OUT/further_hist_$TAG.csv" \
                         "${FURTHER[@]}" > /dev/null
        "$B/pbench_rmta" "$SYS" cold strict 1000 "$OUT/further_bench_$TAG.csv" "$OUT/further_hist_$TAG.csv" \
                         "${FURTHER_RMTA[@]}" > /dev/null
      fi
    done
    rm -f "$SYS"; echo "done $TAG"
  done
done
