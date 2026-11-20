# robust-multigrid-reactive-flow

Source code of the MSc thesis

> Ömer Uğur Zayıfoğlu, *Application of the Robust Multigrid Technique to a Structured-Grid
> Reacting-Flow Solver*, Middle East Technical University, Department of Aerospace Engineering, 2026.

The repository contains the codes that produced the results of the thesis:

| Folder | Content | Thesis |
|---|---|---|
| `solver/` | Two-dimensional low-Mach reacting-flow solver (laminar opposed-jet diffusion flame, variable density, single global reaction). Pressure solver: production RMT or geometry-aware RMT (RMTA). C, OpenMP. | Chapters 3 and 5 |
| `include/` | RMT (`rmt_impl.h`) and RMTA (`rmta_impl.h`) pressure solvers, shared by the solver and PBENCH. | Chapter 3 |
| `pbench/` | Pressure benchmark code (PBENCH): solves stored pressure systems with RMT, RMTA and full-depth classical multigrid (pointwise and line smoother). `mms_gen.c` writes the systems of the MMS study. | Chapters 4 and 5 |
| `lfa/` | Local Fourier analysis of the red–black Gauss–Seidel smoother: script and description of the method (`lfa/README.md`). | Chapter 3 |
| `openfoam/` | OpenFOAM v2312 reference case (`rhoReactingFoam`, ideal gas). | Chapter 5 |
| `scripts/` | Scripts that run the cases, the MMS study and the solver comparison of the thesis. | Chapters 4 and 5 |

## Requirements

- C compiler with OpenMP (GCC 12.3 was used), `make`
- Python 3 with NumPy (only for `lfa/`)
- OpenFOAM v2312 with MPI (only for `openfoam/`)

## Build

```
make
```

This builds five programs in `bin/`:

| Program | Description |
|---|---|
| `opposedflow` | reacting-flow solver with production RMT |
| `opposedflow_rmta` | the same solver with RMTA (`-DRMT_ANISO`) |
| `pbench` | pressure benchmark: RMT and classical multigrid |
| `pbench_rmta` | pressure benchmark: RMTA (`-DRMT_ANISO`) |
| `mms_gen` | writes a system of the MMS study |

## Quick start

Nominal case on the coarsest mesh M1 (108 × 36 cells) to t = 2 s:

```
OMP_NUM_THREADS=8 scripts/run_reacting.sh NOM M1 2.001
```

One system of the MMS study (aspect ratio 3, M1) with three solvers:

```
bin/mms_gen 108 36 0.02 0.02 mms.bin
PBENCH_MMS_OUT=mms_error.csv bin/pbench mms.bin cold strict 1000 bench.csv hist.csv \
    "RMT:s=16" "MG:r=2:g=1:L=0:c=direct" "MG:r=2:g=1:L=0:c=direct:sm=line"
bin/pbench_rmta mms.bin cold strict 1000 bench.csv hist.csv "RMTA:s=6"
```

The cycle counts are 8 (RMT), 17 (MG2-V), 10 (MG2-V with line smoother) and 10 (RMTA, s = 6),
as in the cycle-count tables of Chapter 4.

## Reproducing the results of the thesis

The thesis results were computed on the Orfoz cluster of TÜBİTAK ULAKBİM TRUBA
(2 × Intel Xeon Platinum 8480+ per node) with 56 OpenMP threads on one socket.
The cycle counts do not depend on the number of threads; the wall times depend on the hardware.

### Smoothing factors (Chapter 3)

```
python3 lfa/lfa_rbgs.py
```

### MMS study (Chapter 4)

```
scripts/run_mms_study.sh results/mms 1 16 56
```

Aspect ratios 1, 3, 9 and 27 on meshes M1–M6, all ten solver configurations, zero initial
guess, strict test ‖r‖∞ ≤ 10⁻¹⁰ ‖r₀‖∞. The discretisation errors are written to
`mms_error_*.csv`, the cycle counts and times to `bench_*.csv`, and the residual histories to
`hist_*.csv`.

### Reacting-flow runs (Chapter 5)

`scripts/run_reacting.sh CASE MESH END [rmt|rmta] [SWEEPS]` runs one case. The cases are

| CASE | Description |
|---|---|
| `NOM` | nominal case (inflow speed 0.10 m/s, A = 10⁷, β = 5, T_a = 1000 K, Pr = 0.7) |
| `L1` | as NOM with Pr = Sc = 1 (unity Lewis number, Burke–Schumann test) |
| `BC1`, `BC3` | inflow speed 0.05 and 0.20 m/s |
| `S2`, `S3` | A = 10⁸, T_a = 500 K and A = 10⁹, T_a = 100 K |

and the meshes are M1 (108 × 36) to M6 (3456 × 1152), all with Δy = 3Δx.
The runs of the thesis (table of runs in Chapter 5) are

| Runs | Command |
|---|---|
| nominal case to the steady state, M1–M4 | `scripts/run_reacting.sh NOM M1 8.001` (and M2, M3, M4) |
| unity Lewis number, M2–M4 | `scripts/run_reacting.sh L1 M2 8.001` (and M3, M4) |
| chemistry cases, M2–M4 | `scripts/run_reacting.sh S2 M2 2.001` (and S3; M3, M4) |
| flow-intensity cases, M1–M4 | `scripts/run_reacting.sh BC1 M1 2.001` (and BC3; M2–M4) |
| nominal case on M5 | `scripts/run_reacting.sh NOM M5 2.001` |
| nominal case on M6 | `scripts/run_reacting.sh NOM M6 0.1001 rmta 6` |
| in-solver comparison, M1–M4 | `scripts/run_reacting.sh NOM M1 2.001 rmt 16`, `... rmta 6`, `... rmta 4` |

**Recorded pressure systems.** The solver writes the pressure systems (right-hand side and
initial guesses) used by PBENCH when `PDUMP_STEPS` or `PDUMP_TIMES` is set. The thesis used

```
PDUMP_STEPS=1,2,3,6,12,100,1000,10000
PDUMP_TIMES=0.1,0.2,0.3,0.4,0.5,0.75,1,1.5,2        # and 4,8 on M1-M4
```

**Grid sequencing (systems on M5 and M6).** A run with checkpoints (`CKPT_TIMES=...`) on M4
is continued on the finer mesh from a checkpoint at time t₀. The state is interpolated
bilinearly, and the systems at t₀ + 1 ms and t₀ + 2 ms are recorded, for example

```
CKPT_TIMES=0.2,0.3,0.4,0.5,0.75,1,1.5,2,4,8 scripts/run_reacting.sh NOM M4 8.001
RESTART=runs/NOM_M4_rmt16/ckpt/ckpt_864x288_t0.5.bin PDUMP_TIMES=0.501,0.502 VDC_NO_VTI=1 \
    scripts/run_reacting.sh NOM M6 0.5021 rmta 6
```

### Pressure-solver comparison on the recorded systems (Chapter 5)

```
THREADS="1 16 56 112" scripts/run_pbench.sh results/pbench runs/*/dumps
```

Every system is solved by the ten configurations with five combinations of initial guess
(recorded, uniform p₀, zero) and stopping test (production, strict).

### OpenFOAM reference (Chapter 5)

```
cd openfoam/opposed_jet_perfectGas
./Allrun M2 16            # 0 to 2 s on M2 with 16 MPI processes (also M3, M4)
./continue_to_8s.sh 16    # continues the finished run to 8 s
```

The fields along the centerline x = L_x/2 are sampled with `postProcess -func sample`
(`system/sampleDict`).

## Solver configurations

| Name in the thesis | PBENCH solver string |
|---|---|
| RMT (production, s = 16) | `RMT:s=16` |
| RMTA (s = 4, s = 6) | `RMTA:s=4`, `RMTA:s=6` (with `pbench_rmta`) |
| MG2-V, MG2-W, MG3-V | `MG:r=2:g=1:L=0:c=direct`, `MG:r=2:g=2:L=0:c=direct`, `MG:r=3:g=1:L=0:c=direct` |
| MG2-V line (3,3) | `MG:r=2:g=1:L=0:c=direct:sm=line` |
| MG2-V line (1,1) | `MG:r=2:g=1:L=0:c=direct:sm=line:pre=1:post=1` |
| MG2-W line (1,1) | `MG:r=2:g=2:L=0:c=direct:sm=line:pre=1:post=1` |
| MG3-V line (3,3) | `MG:r=3:g=1:L=0:c=direct:sm=line` |

The classical multigrid solvers go down to the coarsest admissible grid and solve it with a
banded Cholesky factorisation; they use three pre- and three post-smoothing sweeps unless
`pre` and `post` are given.

## Citation

See `CITATION.cff`.

## License

MIT, see `LICENSE`.
