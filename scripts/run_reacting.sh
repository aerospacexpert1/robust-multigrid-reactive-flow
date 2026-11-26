#!/usr/bin/env bash
# One reacting-flow run of the thesis (Chapter 5).
# Usage: scripts/run_reacting.sh CASE MESH END_TIME [SOLVER] [SWEEPS] [OUTDIR]
#   CASE    NOM (nominal, = BC2 = S1), L1 (Pr = Sc = 1, unity Lewis number), BC1, BC3, S2, S3,
#           HQ2, HQP (nominal case with twice the heat of reaction and with the physical value)
#   MESH    M1 (108x36) ... M6 (3456x1152), all with dy = 3 dx
#   SOLVER  rmt (default, production RMT) or rmta (geometry-aware RMT)
#   SWEEPS  smoothing sweeps s (default 16 for rmt, 6 for rmta)
# Threads: OMP_NUM_THREADS (the thesis used 56 threads on one socket).
# Pressure systems for PBENCH: set PDUMP_STEPS / PDUMP_TIMES (see README); checkpoints: CKPT_TIMES.
# RESTART=checkpoint continues a run (on a finer mesh with bilinear interpolation); EXTRA_ARGS adds
# solver options, e.g. EXTRA_ARGS="-maxCo 0.025 -maxDi 0.225" halves the time step (time-step check).
# Examples:
#   scripts/run_reacting.sh NOM M2 8.001               # nominal case to the steady state
#   PDUMP_STEPS=1,2,3,6,12,100,1000,10000 PDUMP_TIMES=0.1,0.2,0.3,0.4,0.5,0.75,1,1.5,2 \
#     scripts/run_reacting.sh BC1 M3 2.001             # with recorded pressure systems
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CASE=${1:?CASE}; MESH=${2:?MESH}; END=${3:?END_TIME}; SOLVER=${4:-rmt}
case $SOLVER in rmt) BIN=$ROOT/bin/opposedflow; SW=${5:-16};; rmta) BIN=$ROOT/bin/opposedflow_rmta; SW=${5:-6};;
  *) echo "SOLVER must be rmt or rmta" >&2; exit 2;; esac
OUT=${6:-runs/${CASE}_${MESH}_${SOLVER}${SW}}
declare -A NX=([M1]=108 [M2]=216 [M3]=432 [M4]=864 [M5]=1728 [M6]=3456)
declare -A NY=([M1]=36  [M2]=72  [M3]=144 [M4]=288 [M5]=576  [M6]=1152)
declare -A CASEARGS=(
 [NOM]="-vFuel 0.10 -vOx -0.10 -Pr 0.7 -Sc 1.0 -A 1e7 -beta 5 -Ta 1000"
 [L1]="-vFuel 0.10 -vOx -0.10 -Pr 1.0 -Sc 1.0 -A 1e7 -beta 5 -Ta 1000"
 [BC1]="-vFuel 0.05 -vOx -0.05 -Pr 0.7 -Sc 1.0 -A 1e7 -beta 5 -Ta 1000"
 [BC3]="-vFuel 0.20 -vOx -0.20 -Pr 0.7 -Sc 1.0 -A 1e7 -beta 5 -Ta 1000"
 [S2]="-vFuel 0.10 -vOx -0.10 -Pr 0.7 -Sc 1.0 -A 1e8 -beta 5 -Ta 500"
 [S3]="-vFuel 0.10 -vOx -0.10 -Pr 0.7 -Sc 1.0 -A 1e9 -beta 5 -Ta 100"
 [HQ2]="-vFuel 0.10 -vOx -0.10 -Pr 0.7 -Sc 1.0 -A 1e7 -beta 5 -Ta 1000 -HfF 7.142857143e6"
 [HQP]="-vFuel 0.10 -vOx -0.10 -Pr 0.7 -Sc 1.0 -A 1e7 -beta 5 -Ta 1000 -HfF 1.010714286e7")
[[ -n "${NX[$MESH]:-}" ]] || { echo "unknown mesh $MESH" >&2; exit 2; }
[[ -n "${CASEARGS[$CASE]:-}" ]] || { echo "unknown case $CASE" >&2; exit 2; }
TH=${OMP_NUM_THREADS:-$(nproc)}
COMMON="-threads $TH -dt 1e-7 -dtMax 1e-4 -rmtLevels 0 -rmtOmega 1 -pCycles 500 -pRelTol 1e-4 -pAbsTol 1e-6 \
 -sCycles 4 -sSweeps 5 -writeOutput 1 -progressEvery 20000 -variableCp 0 -sutherland 0 -rhoRelax 1 \
 -mu 2.0e-5 -cp 1000.0 -HfF 3.571428571e6 -lowMachDiag 1 -perfectGas 1"
mkdir -p "$OUT"
[[ -n "${PDUMP_STEPS:-}${PDUMP_TIMES:-}" ]] && { export PDUMP_DIR=${PDUMP_DIR:-$OUT/dumps}; mkdir -p "$PDUMP_DIR"; }
[[ -n "${CKPT_TIMES:-}" ]] && { export CKPT_DIR=${CKPT_DIR:-$OUT/ckpt}; mkdir -p "$CKPT_DIR"; }
export OMP_NUM_THREADS=$TH OMP_DYNAMIC=false OMP_PROC_BIND=${OMP_PROC_BIND:-close} OMP_PLACES=${OMP_PLACES:-cores}
echo "run $CASE $MESH (${NX[$MESH]}x${NY[$MESH]}) to t=$END with $SOLVER s=$SW, $TH threads -> $OUT"
"$BIN" -Nx "${NX[$MESH]}" -Ny "${NY[$MESH]}" $COMMON ${CASEARGS[$CASE]} -rmtSweeps "$SW" -end "$END" \
  -write "${WRITE:-1.0}" -caseId "${CASE}_${MESH}" -outputDir "$OUT" ${RESTART:+-restart "$RESTART"} \
  ${EXTRA_ARGS:-} > "$OUT/run.log" 2>&1
tail -3 "$OUT/run.log"
