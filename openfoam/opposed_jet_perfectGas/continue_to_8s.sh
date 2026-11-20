#!/usr/bin/env bash
# Continues a finished run (latest time 2) to t = 8 s, writing every 1 s.
# Usage: ./continue_to_8s.sh [NPROCS]      (default 16; any process count can be used)
set -euo pipefail
cd "$(dirname "$0")"
NPROCS=${1:-16}
foamDictionary -entry startFrom     -set latestTime system/controlDict > /dev/null
foamDictionary -entry endTime       -set 8.0        system/controlDict > /dev/null
foamDictionary -entry writeInterval -set 1.0        system/controlDict > /dev/null
if [[ "$NPROCS" -gt 1 ]]; then
    foamDictionary -entry numberOfSubdomains -set "$NPROCS" system/decomposeParDict > /dev/null
    foamDictionary -entry simpleCoeffs/n -set "($NPROCS 1 1)" system/decomposeParDict > /dev/null
    decomposePar -force -latestTime > log.decomposePar_t8 2>&1
    mpirun -np "$NPROCS" rhoReactingFoam -parallel > log.rhoReactingFoam_t8 2>&1
    reconstructPar -newTimes > log.reconstructPar_t8 2>&1
else
    rhoReactingFoam > log.rhoReactingFoam_t8 2>&1
fi
echo "finished: latest time $(foamListTimes -latestTime 2>/dev/null | tail -1)"
