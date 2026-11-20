/*
  opposedflow.c -- two-dimensional low-Mach reacting-flow solver (laminar opposed-jet
  diffusion flame) used in the thesis
  "Application of the Robust Multigrid Technique to a Structured-Grid Reacting-Flow Solver".

  Model and numerics (thesis Chapter 3):
    - uniform structured cell-centred finite-volume mesh, Nx x Ny cells;
    - variable density from the ideal-gas law (-perfectGas 1), constant transport
      properties, single global reaction 2F + O -> 2P with Arrhenius rate A T^beta exp(-Ta/T);
    - explicit predictor and non-incremental projection on the mass flux m = rho u
      (one pressure solve per time step); the pressure operator is the constant-coefficient
      five-point Laplacian;
    - pressure solver: production RMT (include/rmt_impl.h) or, with -DRMT_ANISO,
      geometry-aware RMT (include/rmta_impl.h); OpenMP parallel.

  Build (see the top-level Makefile):
    gcc -O3 -march=native -std=c11 -fopenmp -I../include opposedflow.c -lm -o opposedflow
    gcc -O3 -march=native -std=c11 -fopenmp -I../include -DRMT_ANISO opposedflow.c -lm -o opposedflow_rmta

  Example (nominal case on M1, see scripts/run_reacting.sh for all cases of the thesis):
    OMP_NUM_THREADS=8 ./opposedflow -Nx 108 -Ny 36 -end 2.0 -threads 8 -perfectGas 1 \
      -vFuel 0.10 -vOx -0.10 -A 1e7 -beta 5 -Ta 1000 -rmtSweeps 16 -outputDir out

  Optional environment variables:
    PDUMP_STEPS, PDUMP_TIMES, PDUMP_DIR  write the pressure systems (right-hand side and
                                         initial guesses) at the given steps or times;
                                         these files are the input of pbench/pbench.c
    CKPT_TIMES, CKPT_DIR                 write restart checkpoints at the given times
    VDC_NO_VTI=1                         no VTI field output
  -restart FILE continues from a checkpoint; on a finer mesh the state is interpolated
  bilinearly (grid sequencing, thesis Chapter 5).
*/

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <time.h>
#include <limits.h>
#include <omp.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define IDX(i,j,ny) ((i)*((ny)+2) + (j))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))
#if defined(__GNUC__) || defined(__clang__)
#define MAYBE_UNUSED __attribute__((unused))
#else
#define MAYBE_UNUSED
#endif

typedef struct {
    int Nx, Ny;
    double Lx, Ly;
    double dx, dy;
    double x1, x2;
} Grid;

typedef struct {
    const char *name;
    double W;
    double Tlow, Thigh, Tcommon;
    double lowCpCoeffs[7];
    double highCpCoeffs[7];
    double hRef;
    double As, Ts;
} JanafSpecies;

typedef struct {
    double rho, mu, cp, Pr, Sc;
    double nu, alpha, D;
    bool variableRhoOn;
    bool sutherlandOn;
    bool hydroPressureInDensityOn;
    double rhoRelax;

    double W_F, W_O, W_P, W_N2;
    double Hf_F, Hf_O, Hf_P, Hf_N2;
    double A, beta, Ta;

    double T_in;
    double vFuel, vOx;
    double YF_fuel, YO_fuel, YP_fuel, YN2_fuel;
    double YF_ox,   YO_ox,   YP_ox,   YN2_ox;

    bool chemistryOn;
    double initialChemicalTimeStep;
    double maxChemicalTimeStep;
    double dTchemMax;
    double reactedFracMax;
    double p0; /* absolute total pressure at outlet */
    double Tref; /* sensible enthalpy reference temperature */
    bool variableCpOn;
    JanafSpecies spF, spO, spP, spN2;
} Phys;

typedef struct {
    double startTime, endTime, dt0, writeInterval;
    double maxCo, maxDi, dtMax;
    int nOuterCorrectors, nCorrectors; /* kept for command-line compatibility; the scheme requires 1 */
    int poissonIters;                  /* maximum RMT cycles per pressure solve */
    int scalarIters;                   /* scalar cycles */
    int scalarSweepsPerCycle;

    /* RMT pressure relaxation. The mg* options are kept only for command-line compatibility. */
    int mgPreSmooth;
    int mgPostSmooth;
    int mgCoarseSweeps;
    double mgOmega;

    double pressureRelTol, pressureAbsTol;
    double relaxU, relaxSc, relaxH;
    int progressEvery;
    int threads;
    bool writeOutput;
    char outputDir[PATH_MAX];
    char caseId[256];
    int lowMachDiag; /* LOWMACH: 1 -> write thermal-expansion diagnostics (no effect on the solution) */
    char restartFile[PATH_MAX]; /* checkpoint to start from ("" = start at t=0) */
} Controls;

typedef struct {
    double *p, *u, *v, *T, *h, *YF, *YO, *YP, *YN2;
    double *rho, *rhoOld, *muMix, *nuMix, *alphaMix, *DMix, *cpMix;
    double *us, *vs, *rhs, *work, *phiOld, *rhsScalar;
    double *ufx, *vfy, *mfx, *mfy; /* collocated face-normal volume and mass fluxes */
} Fields;

typedef struct {
    int cycles;
    double rhsNorm;
    double initialResidual;
    double absResidual;
    double relResidual;
    double convergenceFactor;
} PressureStats;


typedef struct {
    int count;
    int capacity;
    double *absResidual;
    double *relResidual;
} CycleHistory;

typedef struct {
    int step;
    double time;
    double dt;
    int cycles;
    double initialResidual;
    double finalResidual;
    double finalRelativeResidual;
    double convergenceFactor;
} PressureStepRecord;

typedef struct {
    PressureStepRecord *data;
    size_t count;
    size_t capacity;
} PressureStepHistory;

typedef struct {
    double maxAbs;
    double l2;
    double integral;
} MassStats;

typedef struct {
    FILE *fp;
    char path[PATH_MAX];
} PvdWriter;

static double u_face_x(const Grid *g, const Fields *f, int iface, int j);
static double v_face_y(const Grid *g, const Phys *ph, const Fields *f, int i, int jface);
static double rho_face_x(const Grid *g, const Fields *f, int iface, int j);
static double rho_face_y(const Grid *g, const Fields *f, int i, int jface);
static double face_avg_x_coeff(const Grid *g, const double *a, int i, int j);
static double face_avg_y_coeff(const Grid *g, const double *a, int i, int j);


static void die(const char *msg) { fprintf(stderr, "FATAL: %s\n", msg); exit(1); }
static double clamp(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }
static int ensure_dir(const char *path) {
    if (mkdir(path, 0777) == 0 || errno == EEXIST) return 0;
    return -1;
}

static int ensure_dir_recursive(const char *path) {
    if (!path || !*path) return -1;
    char tmp[PATH_MAX];
    size_t len = strlen(path);
    if (len >= sizeof(tmp)) return -1;
    memcpy(tmp, path, len + 1);
    while (len > 1 && tmp[len-1] == '/') tmp[--len] = '\0';
    for (char *q = tmp + 1; *q; ++q) {
        if (*q == '/') {
            *q = '\0';
            if (*tmp && ensure_dir(tmp) != 0) return -1;
            *q = '/';
        }
    }
    return ensure_dir(tmp);
}

static void path_join(char *dst, size_t n, const char *a, const char *b) {
    int nw = snprintf(dst, n, "%s%s%s", a, (a[0] && a[strlen(a)-1] == '/') ? "" : "/", b);
    if (nw < 0 || (size_t)nw >= n) die("output path too long");
}


static const double RU_J_PER_KMOLK = 8314.46261815324;

static void set_janaf_species(JanafSpecies *sp, const char *name, double W, double Tlow, double Thigh, double Tcommon,
                              const double low[7], const double high[7], double As, double Ts) {
    sp->name = name;
    sp->W = W;
    sp->Tlow = Tlow;
    sp->Thigh = Thigh;
    sp->Tcommon = Tcommon;
    for (int k=0; k<7; ++k) {
        sp->lowCpCoeffs[k] = low[k];
        sp->highCpCoeffs[k] = high[k];
    }
    sp->hRef = 0.0;
    sp->As = As;
    sp->Ts = Ts;
}

static inline double janaf_limit_T(const JanafSpecies *sp, double T) {
    return clamp(T, sp->Tlow, sp->Thigh);
}

static inline const double *janaf_coeffs(const JanafSpecies *sp, double T) {
    double Tc = janaf_limit_T(sp, T);
    return (Tc < sp->Tcommon) ? sp->lowCpCoeffs : sp->highCpCoeffs;
}

static double janaf_Cp_mass(const JanafSpecies *sp, double T) {
    T = janaf_limit_T(sp, T);
    const double *a = janaf_coeffs(sp, T);
    return (RU_J_PER_KMOLK / sp->W) * ((((a[4]*T + a[3])*T + a[2])*T + a[1])*T + a[0]);
}

static double janaf_Ha_mass(const JanafSpecies *sp, double T) {
    T = janaf_limit_T(sp, T);
    const double *a = janaf_coeffs(sp, T);
    return (RU_J_PER_KMOLK / sp->W) * (((((a[4]/5.0*T + a[3]/4.0)*T + a[2]/3.0)*T + a[1]/2.0)*T + a[0])*T + a[5]);
}

static void init_janaf_defaults(Phys *ph) {
    const double AsCO = 1.663e-06, TsCO = 136.0;
    const double AsO2 = 1.6934e-06, TsO2 = 127.0;
    const double AsCO2 = 1.370e-06, TsCO2 = 222.0;
    const double AsN2 = 1.406e-06, TsN2 = 111.0;

    static const double CO_low[7]  = { 3.57953347, -0.00061035368,  1.01681433e-06,  9.07005884e-10, -9.04424499e-13, -14344.0860,   3.50840928 };
    static const double CO_high[7] = { 2.71518561,  0.00206252743, -9.98825771e-07,  2.30053008e-10, -2.03647716e-14, -14151.8724,   7.81868772 };

    static const double O2_low[7]  = { 3.78245636, -0.00299673416,  9.84730201e-06, -9.68129509e-09,  3.24372837e-12,  -1063.94356,  3.65767573 };
    static const double O2_high[7] = { 3.28253784,  0.00148308754, -7.57966669e-07,  2.09470555e-10, -2.16717794e-14,  -1088.45772,  5.45323129 };

    static const double CO2_low[7]  = { 2.35677352,  0.00898459677, -7.12356269e-06,  2.45919022e-09, -1.43699548e-13, -48371.9697,  9.90105222 };
    static const double CO2_high[7] = { 3.85746029,  0.00441437026, -2.21481404e-06,  5.23490188e-10, -4.72084164e-14, -48759.1660,  2.27163806 };

    static const double N2_low[7]  = { 3.29867700,  0.00140824040, -3.96322200e-06,  5.64151500e-09, -2.44485400e-12,  -1020.89990,  3.95037200 };
    static const double N2_high[7] = { 2.92664000,  0.00148797680, -5.68476000e-07,  1.00970380e-10, -6.75335100e-15,   -922.79770,  5.98052800 };

    set_janaf_species(&ph->spF,  "F/CO",  28.01055, 200.0, 3500.0, 1000.0, CO_low,  CO_high,  AsCO,  TsCO);
    set_janaf_species(&ph->spO,  "O/O2",  31.99880, 200.0, 3500.0, 1000.0, O2_low,  O2_high,  AsO2,  TsO2);
    set_janaf_species(&ph->spP,  "P/CO2", 44.00995, 200.0, 3500.0, 1000.0, CO2_low, CO2_high, AsCO2, TsCO2);
    set_janaf_species(&ph->spN2, "N2",    28.01340, 250.0, 5000.0, 1000.0, N2_low,  N2_high,  AsN2,  TsN2);
}

static void update_janaf_references(Phys *ph) {
    ph->spF.hRef  = janaf_Ha_mass(&ph->spF,  ph->Tref);
    ph->spO.hRef  = janaf_Ha_mass(&ph->spO,  ph->Tref);
    ph->spP.hRef  = janaf_Ha_mass(&ph->spP,  ph->Tref);
    ph->spN2.hRef = janaf_Ha_mass(&ph->spN2, ph->Tref);
}

static inline double species_h_sensible_rel(const JanafSpecies *sp, double T) {
    return janaf_Ha_mass(sp, T) - sp->hRef;
}

static inline double mixture_cp(const Phys *ph, double YF, double YO, double YP, double YN2, double T) {
    if (!ph->variableCpOn) return ph->cp;
    return YF*janaf_Cp_mass(&ph->spF, T) + YO*janaf_Cp_mass(&ph->spO, T) + YP*janaf_Cp_mass(&ph->spP, T) + YN2*janaf_Cp_mass(&ph->spN2, T);
}

static inline double mixture_h_sensible_rel(const Phys *ph, double YF, double YO, double YP, double YN2, double T) {
    if (!ph->variableCpOn) return ph->cp*(T - ph->Tref);
    return YF*species_h_sensible_rel(&ph->spF, T) + YO*species_h_sensible_rel(&ph->spO, T) + YP*species_h_sensible_rel(&ph->spP, T) + YN2*species_h_sensible_rel(&ph->spN2, T);
}

static inline double mixture_mw(const Phys *ph, double YF, double YO, double YP, double YN2) {
    double invW = YF/ph->W_F + YO/ph->W_O + YP/ph->W_P + YN2/ph->W_N2;
    return (invW > 1.0e-16) ? 1.0/invW : ph->W_N2;
}

static inline double mixture_R_mass(const Phys *ph, double YF, double YO, double YP, double YN2) {
    return RU_J_PER_KMOLK / mixture_mw(ph, YF, YO, YP, YN2);
}

static inline double sutherland_mu_species(const JanafSpecies *sp, double T) {
    T = MAX(T, 150.0);
    return sp->As*sqrt(T)/(1.0 + sp->Ts/T);
}

static inline double mixture_mu(const Phys *ph, double YF, double YO, double YP, double YN2, double T) {
    if (!ph->sutherlandOn) return ph->mu;
    return YF*sutherland_mu_species(&ph->spF, T)
         + YO*sutherland_mu_species(&ph->spO, T)
         + YP*sutherland_mu_species(&ph->spP, T)
         + YN2*sutherland_mu_species(&ph->spN2, T);
}

static inline double mixture_density(const Phys *ph, double YF, double YO, double YP, double YN2, double T, double pAbs) {
    if (!ph->variableRhoOn) return ph->rho;
    double Rmix = mixture_R_mass(ph, YF, YO, YP, YN2);
    double pUse = MAX(pAbs, 1.0e3);
    return pUse / MAX(Rmix*MAX(T, 200.0), 1.0);
}

static double T_from_h_mix(const Phys *ph, double YF, double YO, double YP, double YN2, double h, double Tguess) {
    if (!ph->variableCpOn) return ph->Tref + h/ph->cp;
    double Tmin = 250.0, Tmax = 5000.0;
    if (Tguess < Tmin || Tguess > Tmax || !isfinite(Tguess)) Tguess = ph->Tref + h / MAX(ph->cp, 1.0);
    double hLo = mixture_h_sensible_rel(ph, YF, YO, YP, YN2, Tmin);
    double hHi = mixture_h_sensible_rel(ph, YF, YO, YP, YN2, Tmax);
    if (h <= hLo) return Tmin;
    if (h >= hHi) return Tmax;
    double T = clamp(Tguess, Tmin, Tmax);
    for (int it=0; it<12; ++it) {
        double hEval = mixture_h_sensible_rel(ph, YF, YO, YP, YN2, T);
        double cpMix = MAX(mixture_cp(ph, YF, YO, YP, YN2, T), 1.0);
        double dT = (hEval - h) / cpMix;
        T -= dT;
        if (T <= Tmin || T >= Tmax || !isfinite(T)) break;
        if (fabs(dT) < 1.0e-10*MAX(1.0, fabs(T))) return T;
    }
    double lo = Tmin, hi = Tmax;
    for (int it=0; it<60; ++it) {
        double mid = 0.5*(lo + hi);
        double hMid = mixture_h_sensible_rel(ph, YF, YO, YP, YN2, mid);
        if (hMid < h) lo = mid; else hi = mid;
    }
    return 0.5*(lo + hi);
}

static inline double h_from_T_mix(const Phys *ph, double YF, double YO, double YP, double YN2, double T) {
    return mixture_h_sensible_rel(ph, YF, YO, YP, YN2, T);
}


static void update_thermo_transport_fields(const Grid *g, const Phys *ph, Fields *f, bool relaxDensity) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1; i<=g->Nx; ++i) for (int j=1; j<=g->Ny; ++j) {
        double yF = clamp(f->YF[IDX(i,j,g->Ny)], 0.0, 1.0);
        double yO = clamp(f->YO[IDX(i,j,g->Ny)], 0.0, 1.0);
        double yP = clamp(f->YP[IDX(i,j,g->Ny)], 0.0, 1.0);
        double yN2 = MAX(0.0, 1.0 - yF - yO - yP);
        double T = MAX(f->T[IDX(i,j,g->Ny)], 200.0);
        double cpMix = mixture_cp(ph, yF, yO, yP, yN2, T);
        double pUse = ph->hydroPressureInDensityOn ? f->p[IDX(i,j,g->Ny)] : ph->p0;
        double rhoRaw = mixture_density(ph, yF, yO, yP, yN2, T, pUse);
        double rhoOld = f->rho[IDX(i,j,g->Ny)];
        double rhoMix = rhoRaw;
        if (relaxDensity && ph->variableRhoOn && isfinite(rhoOld) && rhoOld > 0.0 && ph->rhoRelax < 0.999999) {
            rhoMix = ph->rhoRelax*rhoRaw + (1.0 - ph->rhoRelax)*rhoOld;
        }
        double muMix = mixture_mu(ph, yF, yO, yP, yN2, T);
        f->YN2[IDX(i,j,g->Ny)] = yN2;
        f->cpMix[IDX(i,j,g->Ny)] = cpMix;
        f->rho[IDX(i,j,g->Ny)] = rhoMix;
        f->muMix[IDX(i,j,g->Ny)] = muMix;
        f->nuMix[IDX(i,j,g->Ny)] = muMix / MAX(rhoMix, 1.0e-12);
        f->alphaMix[IDX(i,j,g->Ny)] = muMix / MAX(rhoMix*ph->Pr, 1.0e-12);
        f->DMix[IDX(i,j,g->Ny)] = muMix / MAX(rhoMix*ph->Sc, 1.0e-12);
    }
}

static MAYBE_UNUSED double interior_average(const Grid *g, const double *a) {
    double s = 0.0; int n = 0;
    for (int i=1; i<=g->Nx; ++i) for (int j=1; j<=g->Ny; ++j) { s += a[IDX(i,j,g->Ny)]; ++n; }
    return (n > 0) ? s/(double)n : 0.0;
}

static inline double reaction_heat_release_per_extent(const Phys *ph, double T) {
    if (ph->variableCpOn) {
        return 2.0*ph->spF.W*janaf_Ha_mass(&ph->spF, T) + 1.0*ph->spO.W*janaf_Ha_mass(&ph->spO, T) - 2.0*ph->spP.W*janaf_Ha_mass(&ph->spP, T);
    }
    return (2.0*ph->W_F*ph->Hf_F + ph->W_O*ph->Hf_O - 2.0*ph->W_P*ph->Hf_P);
}

static void refresh_temperature_from_enthalpy(const Grid *g, const Phys *ph, Fields *f) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1; i<=g->Nx; ++i) for (int j=1; j<=g->Ny; ++j) {
        double yF = clamp(f->YF[IDX(i,j,g->Ny)], 0.0, 1.0);
        double yO = clamp(f->YO[IDX(i,j,g->Ny)], 0.0, 1.0);
        double yP = clamp(f->YP[IDX(i,j,g->Ny)], 0.0, 1.0);
        double yN2 = MAX(0.0, 1.0 - yF - yO - yP);
        f->YN2[IDX(i,j,g->Ny)] = yN2;
        f->T[IDX(i,j,g->Ny)] = T_from_h_mix(ph, yF, yO, yP, yN2, f->h[IDX(i,j,g->Ny)], f->T[IDX(i,j,g->Ny)]);
    }
}

static void init_case(Grid *g, Phys *ph, Controls *c) {
    g->Nx = 110; g->Ny = 40;
    g->Lx = 0.02; g->Ly = 0.02;
    g->dx = g->Lx / g->Nx; g->dy = g->Ly / g->Ny;
    g->x1 = 5.0*g->Lx/11.0; g->x2 = 6.0*g->Lx/11.0;

    ph->rho = 1.0; ph->mu = 2.0e-5; ph->cp = 1000.0; ph->Pr = 0.7; ph->Sc = 1.0;
    ph->variableRhoOn = true; ph->sutherlandOn = true;
    ph->hydroPressureInDensityOn = false; ph->rhoRelax = 0.35;
    ph->nu = ph->mu/ph->rho; ph->alpha = ph->nu/ph->Pr; ph->D = ph->nu;
    init_janaf_defaults(ph);
    ph->W_F = ph->spF.W; ph->W_O = ph->spO.W; ph->W_P = ph->spP.W; ph->W_N2 = ph->spN2.W;
    ph->Hf_F = 1.0e8 / ph->W_F; ph->Hf_O = 0.0; ph->Hf_P = 0.0; ph->Hf_N2 = 0.0;
    ph->A = 1.0e9; ph->beta = 5.0; ph->Ta = 100.0;
    ph->T_in = 293.0; ph->Tref = ph->T_in; ph->vFuel = +0.1; ph->vOx = -0.1;
    ph->variableCpOn = true;
    update_janaf_references(ph);
    ph->YF_fuel = 1.0; ph->YO_fuel = 0.0;  ph->YP_fuel = 0.0; ph->YN2_fuel = 0.0;
    ph->YF_ox   = 0.0; ph->YO_ox   = 0.21; ph->YP_ox   = 0.0; ph->YN2_ox   = 0.79;
    ph->chemistryOn = true;
    ph->initialChemicalTimeStep = 1e-6;
    ph->maxChemicalTimeStep     = 1e-4;
    ph->dTchemMax               = 20.0;
    ph->reactedFracMax          = 1.0;
    ph->p0 = 1.0e5;

    c->startTime = 0.0; c->endTime = 1.5; c->dt0 = 1e-7;
    c->writeInterval = 0.1;
    c->maxCo = 0.05; c->maxDi = 0.45; c->dtMax = 1.0e-4;
    c->nOuterCorrectors = 1; c->nCorrectors = 1;
    c->poissonIters = 500000; c->scalarIters = 4; c->scalarSweepsPerCycle = 5;
    c->mgPreSmooth = 0;
    c->mgPostSmooth = 16;
    c->mgCoarseSweeps = 0;
    c->mgOmega = 1.0;
    c->pressureRelTol = 1.0e-4; c->pressureAbsTol = 1.0e-6;
    c->relaxU = 1.0; c->relaxSc = 0.8; c->relaxH = 0.8;
    c->progressEvery = 100;
    c->threads = 1;
    c->writeOutput = true;
    snprintf(c->outputDir, sizeof(c->outputDir), "output");
    snprintf(c->caseId, sizeof(c->caseId), "manual");
    c->lowMachDiag = 0;
    c->restartFile[0] = 0;
}

static void parse_args(int argc, char **argv, Grid *g, Phys *ph, Controls *c) {
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i], "-Nx") && i+1<argc) g->Nx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-Ny") && i+1<argc) g->Ny = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-end") && i+1<argc) c->endTime = atof(argv[++i]);
        else if (!strcmp(argv[i], "-dt") && i+1<argc) c->dt0 = atof(argv[++i]);
        else if (!strcmp(argv[i], "-dtMax") && i+1<argc) c->dtMax = atof(argv[++i]);
        else if (!strcmp(argv[i], "-write") && i+1<argc) c->writeInterval = atof(argv[++i]);
        else if (!strcmp(argv[i], "-writeOutput") && i+1<argc) c->writeOutput = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "-outputDir") && i+1<argc) {
            const char *arg = argv[++i];
            if (snprintf(c->outputDir, sizeof(c->outputDir), "%s", arg) >= (int)sizeof(c->outputDir)) die("outputDir path too long");
        }
        else if (!strcmp(argv[i], "-maxCo") && i+1<argc) c->maxCo = atof(argv[++i]);
        else if (!strcmp(argv[i], "-maxDi") && i+1<argc) c->maxDi = atof(argv[++i]);
        else if (!strcmp(argv[i], "-outerCorr") && i+1<argc) c->nOuterCorrectors = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-corr") && i+1<argc) c->nCorrectors = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-pCycles") && i+1<argc) c->poissonIters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-pRelTol") && i+1<argc) c->pressureRelTol = atof(argv[++i]);
        else if (!strcmp(argv[i], "-pAbsTol") && i+1<argc) c->pressureAbsTol = atof(argv[++i]);
        else if (!strcmp(argv[i], "-sCycles") && i+1<argc) c->scalarIters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-sSweeps") && i+1<argc) c->scalarSweepsPerCycle = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-progressEvery") && i+1<argc) c->progressEvery = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-mgPre") || !strcmp(argv[i], "-rmtLevels")) && i+1<argc) c->mgPreSmooth = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-mgPost") || !strcmp(argv[i], "-rmtPost") || !strcmp(argv[i], "-rmtSweeps")) && i+1<argc) c->mgPostSmooth = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-mgCoarseSweeps") || !strcmp(argv[i], "-rmtCoarseSweeps")) && i+1<argc) c->mgCoarseSweeps = atoi(argv[++i]);
        else if ((!strcmp(argv[i], "-mgOmega") || !strcmp(argv[i], "-rbgsOmega") || !strcmp(argv[i], "-rmtOmega")) && i+1<argc) c->mgOmega = atof(argv[++i]);
        else if (!strcmp(argv[i], "-threads") && i+1<argc) c->threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-caseId") && i+1<argc) {
            const char *arg = argv[++i];
            if (snprintf(c->caseId, sizeof(c->caseId), "%s", arg) >= (int)sizeof(c->caseId)) die("caseId too long");
        }
        else if (!strcmp(argv[i], "-relaxU") && i+1<argc) c->relaxU = atof(argv[++i]);
        else if (!strcmp(argv[i], "-relaxSc") && i+1<argc) c->relaxSc = atof(argv[++i]);
        else if (!strcmp(argv[i], "-relaxH") && i+1<argc) c->relaxH = atof(argv[++i]);
        else if (!strcmp(argv[i], "-lowMachDiag") && i+1<argc) c->lowMachDiag = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-restart") && i+1<argc) snprintf(c->restartFile, sizeof(c->restartFile), "%s", argv[++i]);
        else if (!strcmp(argv[i], "-perfectGas") && i+1<argc) ph->variableRhoOn = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "-variableCp") && i+1<argc) ph->variableCpOn = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "-sutherland") && i+1<argc) ph->sutherlandOn = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "-rho") && i+1<argc) ph->rho = atof(argv[++i]);
        else if (!strcmp(argv[i], "-mu") && i+1<argc) ph->mu = atof(argv[++i]);
        else if (!strcmp(argv[i], "-cp") && i+1<argc) ph->cp = atof(argv[++i]);
        else if (!strcmp(argv[i], "-Pr") && i+1<argc) ph->Pr = atof(argv[++i]);
        else if (!strcmp(argv[i], "-Sc") && i+1<argc) ph->Sc = atof(argv[++i]);
        else if (!strcmp(argv[i], "-HfF") && i+1<argc) ph->Hf_F = atof(argv[++i]);
        else if (!strcmp(argv[i], "-hydroPInRho") && i+1<argc) ph->hydroPressureInDensityOn = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "-rhoRelax") && i+1<argc) ph->rhoRelax = atof(argv[++i]);
        else if (!strcmp(argv[i], "-chemistry") && i+1<argc) ph->chemistryOn = atoi(argv[++i]) != 0;
        else if (!strcmp(argv[i], "-chemDt0") && i+1<argc) ph->initialChemicalTimeStep = atof(argv[++i]);
        else if (!strcmp(argv[i], "-chemDtMax") && i+1<argc) ph->maxChemicalTimeStep = atof(argv[++i]);
        else if (!strcmp(argv[i], "-dTchemMax") && i+1<argc) ph->dTchemMax = atof(argv[++i]);
        else if (!strcmp(argv[i], "-reactedFracMax") && i+1<argc) ph->reactedFracMax = atof(argv[++i]);
        else if (!strcmp(argv[i], "-A") && i+1<argc) ph->A = atof(argv[++i]);
        else if (!strcmp(argv[i], "-beta") && i+1<argc) ph->beta = atof(argv[++i]);
        else if (!strcmp(argv[i], "-Ta") && i+1<argc) ph->Ta = atof(argv[++i]);
        else if (!strcmp(argv[i], "-vFuel") && i+1<argc) ph->vFuel = atof(argv[++i]);
        else if (!strcmp(argv[i], "-vOx") && i+1<argc) ph->vOx = atof(argv[++i]);
        else {
            fprintf(stderr, "WARNING: unknown or incomplete option '%s'\n", argv[i]);
        }
    }
    if (c->nOuterCorrectors != 1 || c->nCorrectors != 1) {
        fprintf(stderr, "WARNING: the scheme uses one physical split and one pressure correction per time step; forcing -outerCorr 1 -corr 1.\n");
        c->nOuterCorrectors = 1;
        c->nCorrectors = 1;
    }
    if (g->Nx < 4 || g->Ny < 4) die("Nx and Ny must both be at least 4");
    if (c->dt0 <= 0.0 || c->dtMax <= 0.0 || c->endTime < c->startTime) die("invalid time controls");
    if (c->poissonIters < 1 || c->scalarIters < 1 || c->scalarSweepsPerCycle < 1) die("solver iteration counts must be positive");
    if (c->mgOmega <= 0.0 || c->mgOmega > 2.0) die("rbgsOmega must be in (0,2]");
    if (c->threads < 1) die("threads must be positive");
    if (c->writeOutput && c->outputDir[0] == '\0') die("outputDir must not be empty when output is enabled");
    ph->rhoRelax = clamp(ph->rhoRelax, 0.0, 1.0);
    if (ph->rho <= 0.0 || ph->mu <= 0.0 || ph->cp <= 0.0 || ph->Pr <= 0.0 || ph->Sc <= 0.0)
        die("rho, mu, cp, Pr and Sc must be positive");
    ph->nu = ph->mu/ph->rho;
    ph->alpha = ph->nu/ph->Pr;
    ph->D = ph->nu/ph->Sc;
    g->dx = g->Lx / g->Nx; g->dy = g->Ly / g->Ny; g->x1 = 5.0*g->Lx/11.0; g->x2 = 6.0*g->Lx/11.0;
}

static void alloc_fields(const Grid *g, Fields *f) {
    size_t n = (size_t)(g->Nx+2) * (size_t)(g->Ny+2);
    #define ALLOC(name) do { f->name=(double*)calloc(n,sizeof(double)); if(!f->name) die("alloc " #name); } while(0)
    ALLOC(p); ALLOC(u); ALLOC(v); ALLOC(T); ALLOC(h); ALLOC(YF); ALLOC(YO); ALLOC(YP); ALLOC(YN2);
    ALLOC(rho); ALLOC(rhoOld); ALLOC(muMix); ALLOC(nuMix); ALLOC(alphaMix); ALLOC(DMix); ALLOC(cpMix);
    ALLOC(us); ALLOC(vs); ALLOC(rhs); ALLOC(work); ALLOC(phiOld); ALLOC(rhsScalar); ALLOC(ufx); ALLOC(vfy); ALLOC(mfx); ALLOC(mfy);
    #undef ALLOC
}
static void free_fields(Fields *f) {
    free(f->p); free(f->u); free(f->v); free(f->T); free(f->h);
    free(f->YF); free(f->YO); free(f->YP); free(f->YN2);
    free(f->rho); free(f->rhoOld); free(f->muMix); free(f->nuMix); free(f->alphaMix); free(f->DMix); free(f->cpMix);
    free(f->us); free(f->vs); free(f->rhs); free(f->work); free(f->phiOld); free(f->rhsScalar); free(f->ufx); free(f->vfy); free(f->mfx); free(f->mfy);
}

static int slot_cell(const Grid *g, int i) {
    double x = (i - 0.5) * g->dx;
    return (x >= g->x1 && x < g->x2);
}

static void init_fields(const Grid *g, const Phys *ph, Fields *f) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1; i<=g->Nx; ++i) for (int j=1; j<=g->Ny; ++j) {
        f->p [IDX(i,j,g->Ny)] = ph->p0;
        f->u [IDX(i,j,g->Ny)] = 0.0;
        f->v [IDX(i,j,g->Ny)] = 0.0;
        f->T [IDX(i,j,g->Ny)] = ph->T_in;
        f->h [IDX(i,j,g->Ny)] = 0.0;
        f->YF[IDX(i,j,g->Ny)] = 0.0;
        f->YO[IDX(i,j,g->Ny)] = 0.0;
        f->YP[IDX(i,j,g->Ny)] = 0.0;
        f->YN2[IDX(i,j,g->Ny)] = 1.0;
        f->rho[IDX(i,j,g->Ny)] = ph->rho;
        f->rhoOld[IDX(i,j,g->Ny)] = ph->rho;
        f->muMix[IDX(i,j,g->Ny)] = ph->mu;
        f->nuMix[IDX(i,j,g->Ny)] = ph->nu;
        f->alphaMix[IDX(i,j,g->Ny)] = ph->alpha;
        f->DMix[IDX(i,j,g->Ny)] = ph->D;
        f->cpMix[IDX(i,j,g->Ny)] = ph->cp;
    }
}

static inline double T_from_h(const Phys *ph, double h) { return ph->Tref + h/ph->cp; }
static inline double h_from_T(const Phys *ph, double T) { return ph->cp*(T - ph->Tref); }

static MAYBE_UNUSED void relax_field(const Grid *g, double *phi, const double *old, double alpha) {
    if (alpha >= 0.999999) return;
    const int nx=g->Nx, ny=g->Ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) {
        double v = alpha*phi[IDX(i,j,ny)] + (1.0-alpha)*old[IDX(i,j,ny)];
        phi[IDX(i,j,ny)] = isfinite(v) ? v : old[IDX(i,j,ny)];
    }
}


/* ---- OpenFOAM-style patch-face approximations on a collocated grid ---- */
static double p_patch_left(const Grid *g, const Phys *ph, const Fields *f, int j) {
    (void)g;
    double uN = f->u[IDX(1,j,g->Ny)];
    if (uN < 0.0) return ph->p0; /* outflow */
    return ph->p0 - 0.5*f->rho[IDX(1,j,g->Ny)]*(uN*uN); /* inflow: use normal component only */
}
static double p_patch_right(const Grid *g, const Phys *ph, const Fields *f, int j) {
    (void)g;
    double uN = f->u[IDX(g->Nx,j,g->Ny)];
    if (uN > 0.0) return ph->p0; /* outflow */
    return ph->p0 - 0.5*f->rho[IDX(g->Nx,j,g->Ny)]*(uN*uN);
}

static void apply_bc_velocity_pressure(const Grid *g, const Phys *ph, Fields *f) {
    const int nx = g->Nx, ny = g->Ny;
    for (int j=1; j<=ny; ++j) {
        /* outlet sides: totalPressure + pressureInletOutletVelocity(value=(0 0)) */
        double pL = p_patch_left(g, ph, f, j), pR = p_patch_right(g, ph, f, j);
        f->p[IDX(0,    j,ny)] = 2.0*pL - f->p[IDX(1, j,ny)];
        f->p[IDX(nx+1, j,ny)] = 2.0*pR - f->p[IDX(nx,j,ny)];

        if (f->u[IDX(1,j,ny)] < 0.0) { /* left outflow */
            f->u[IDX(0,j,ny)] = f->u[IDX(1,j,ny)];
            f->v[IDX(0,j,ny)] = f->v[IDX(1,j,ny)];
        } else { /* left inflow */
            f->u[IDX(0,j,ny)] = -f->u[IDX(1,j,ny)];
            f->v[IDX(0,j,ny)] = -f->v[IDX(1,j,ny)];
        }
        if (f->u[IDX(nx,j,ny)] > 0.0) { /* right outflow */
            f->u[IDX(nx+1,j,ny)] = f->u[IDX(nx,j,ny)];
            f->v[IDX(nx+1,j,ny)] = f->v[IDX(nx,j,ny)];
        } else {
            f->u[IDX(nx+1,j,ny)] = -f->u[IDX(nx,j,ny)];
            f->v[IDX(nx+1,j,ny)] = -f->v[IDX(nx,j,ny)];
        }
    }

    for (int i=1; i<=nx; ++i) {
        int slot = slot_cell(g, i);
        /* p: zeroGradient on fuel/oxidizer/walls */
        f->p[IDX(i,0,    ny)] = f->p[IDX(i,1, ny)];
        f->p[IDX(i,ny+1, ny)] = f->p[IDX(i,ny,ny)];
        if (slot) {
            /* fixedValue U=(0,vIn) -> ghost = 2*value - interior */
            f->u[IDX(i,0,ny)]    = -f->u[IDX(i,1,ny)];
            f->v[IDX(i,0,ny)]    = 2.0*ph->vFuel - f->v[IDX(i,1,ny)];
            f->u[IDX(i,ny+1,ny)] = -f->u[IDX(i,ny,ny)];
            f->v[IDX(i,ny+1,ny)] = 2.0*ph->vOx   - f->v[IDX(i,ny,ny)];
        } else {
            /* no-slip wall */
            f->u[IDX(i,0,ny)]    = -f->u[IDX(i,1,ny)];
            f->v[IDX(i,0,ny)]    = -f->v[IDX(i,1,ny)];
            f->u[IDX(i,ny+1,ny)] = -f->u[IDX(i,ny,ny)];
            f->v[IDX(i,ny+1,ny)] = -f->v[IDX(i,ny,ny)];
        }
    }
}

static void apply_bc_scalar_of(const Grid *g, const Fields *f, double *s,
                               double bottomSlot, double topSlot, double outletInletValue) {
    const int nx = g->Nx, ny = g->Ny;
    for (int j=1; j<=ny; ++j) {
        /* outlet = inletOutlet */
        if (f->u[IDX(1,j,ny)] < 0.0)
            s[IDX(0,j,ny)] = s[IDX(1,j,ny)]; /* outflow zeroGradient */
        else
            s[IDX(0,j,ny)] = 2.0*outletInletValue - s[IDX(1,j,ny)]; /* inflow fixed inletValue */
        if (f->u[IDX(nx,j,ny)] > 0.0)
            s[IDX(nx+1,j,ny)] = s[IDX(nx,j,ny)];
        else
            s[IDX(nx+1,j,ny)] = 2.0*outletInletValue - s[IDX(nx,j,ny)];
    }
    for (int i=1; i<=nx; ++i) {
        int slot = slot_cell(g, i);
        if (slot) {
            s[IDX(i,0,ny)]    = 2.0*bottomSlot - s[IDX(i,1,ny)];
            s[IDX(i,ny+1,ny)] = 2.0*topSlot    - s[IDX(i,ny,ny)];
        } else {
            s[IDX(i,0,ny)]    = s[IDX(i,1,ny)];
            s[IDX(i,ny+1,ny)] = s[IDX(i,ny,ny)];
        }
    }
}

static double limited_linear_psi(double r) { return clamp(MIN(2.0*r, 1.0), 0.0, 1.0); }

static double bounded_between(double val, double a, double b) {
    double lo = a < b ? a : b;
    double hi = a > b ? a : b;
    if (val < lo) return lo;
    if (val > hi) return hi;
    return val;
}

static double scalar_face_x_raw(const Grid *g, const Fields *f, const double *phi,
                            int iface, int j, double inletValue) {
    const int ny = g->Ny, nx = g->Nx;
    if (iface == 0) {
        /* left outlet patch */
        return (f->u[IDX(1,j,ny)] < 0.0) ? phi[IDX(1,j,ny)] : inletValue;
    }
    if (iface == nx) {
        return (f->u[IDX(nx,j,ny)] > 0.0) ? phi[IDX(nx,j,ny)] : inletValue;
    }
    double uf = 0.5*(f->u[IDX(iface,j,ny)] + f->u[IDX(iface+1,j,ny)]);
    const double eps = 1e-30;
    if (uf >= 0.0) {
        double phiUU = phi[IDX(MAX(iface-1,0), j, ny)];
        double phiU  = phi[IDX(iface, j, ny)];
        double phiD  = phi[IDX(iface+1, j, ny)];
        double denom = phiD - phiU;
        double r = (phiU - phiUU) / (fabs(denom) > eps ? denom : copysign(eps, denom + eps));
        double psi = limited_linear_psi(r);
        return phiU + 0.5*psi*(phiD - phiU);
    } else {
        double phiUU = phi[IDX(MIN(iface+2,nx+1), j, ny)];
        double phiU  = phi[IDX(iface+1, j, ny)];
        double phiD  = phi[IDX(iface, j, ny)];
        double denom = phiD - phiU;
        double r = (phiU - phiUU) / (fabs(denom) > eps ? denom : copysign(eps, denom + eps));
        double psi = limited_linear_psi(r);
        return phiU + 0.5*psi*(phiD - phiU);
    }
}

static double scalar_face_y_raw(const Grid *g, const Phys *ph, const Fields *f, const double *phi,
                            int i, int jface, double bottomSlot, double topSlot) {
    (void)ph;
    const int ny = g->Ny;
    if (jface == 0) {
        return slot_cell(g,i) ? bottomSlot : phi[IDX(i,1,ny)];
    }
    if (jface == ny) {
        return slot_cell(g,i) ? topSlot : phi[IDX(i,ny,ny)];
    }
    double vf = 0.5*(f->v[IDX(i,jface,ny)] + f->v[IDX(i,jface+1,ny)]);
    const double eps = 1e-30;
    if (vf >= 0.0) {
        double phiUU = phi[IDX(i, MAX(jface-1,0), ny)];
        double phiU  = phi[IDX(i, jface, ny)];
        double phiD  = phi[IDX(i, jface+1, ny)];
        double denom = phiD - phiU;
        double r = (phiU - phiUU) / (fabs(denom) > eps ? denom : copysign(eps, denom + eps));
        double psi = limited_linear_psi(r);
        return phiU + 0.5*psi*(phiD - phiU);
    } else {
        double phiUU = phi[IDX(i, MIN(jface+2,ny+1), ny)];
        double phiU  = phi[IDX(i, jface+1, ny)];
        double phiD  = phi[IDX(i, jface, ny)];
        double denom = phiD - phiU;
        double r = (phiU - phiUU) / (fabs(denom) > eps ? denom : copysign(eps, denom + eps));
        double psi = limited_linear_psi(r);
        return phiU + 0.5*psi*(phiD - phiU);
    }
}


static double scalar_face_x_species(const Grid *g, const Fields *f, const double *phi,
                                    int iface, int j, double inletValue) {
    double raw = scalar_face_x_raw(g, f, phi, iface, j, inletValue);
    const int ny=g->Ny, nx=g->Nx;
    if (iface == 0) {
        return bounded_between(raw, phi[IDX(1,j,ny)], inletValue);
    }
    if (iface == nx) {
        return bounded_between(raw, phi[IDX(nx,j,ny)], inletValue);
    }
    double a = phi[IDX(iface,   j, ny)];
    double b = phi[IDX(iface+1, j, ny)];
    return bounded_between(raw, a, b);
}

static double scalar_face_y_species(const Grid *g, const Phys *ph, const Fields *f, const double *phi,
                                    int i, int jface, double bottomSlot, double topSlot) {
    double raw = scalar_face_y_raw(g, ph, f, phi, i, jface, bottomSlot, topSlot);
    const int ny=g->Ny;
    if (jface == 0) {
        double faceVal = slot_cell(g,i) ? bottomSlot : phi[IDX(i,1,ny)];
        return bounded_between(raw, phi[IDX(i,1,ny)], faceVal);
    }
    if (jface == ny) {
        double faceVal = slot_cell(g,i) ? topSlot : phi[IDX(i,ny,ny)];
        return bounded_between(raw, phi[IDX(i,ny,ny)], faceVal);
    }
    double a = phi[IDX(i, jface,   ny)];
    double b = phi[IDX(i, jface+1, ny)];
    return bounded_between(raw, a, b);
}

static double scalar_face_x_h(const Grid *g, const Fields *f, const double *phi,
                              int iface, int j) {
    const int ny=g->Ny, nx=g->Nx;
    if (iface == 0)  return phi[IDX(1, j, ny)];
    if (iface == nx) return phi[IDX(nx,j, ny)];
    return scalar_face_x_raw(g, f, phi, iface, j, 0.0);
}

static double scalar_face_y_h(const Grid *g, const Phys *ph, const Fields *f, const double *phi,
                              int i, int jface) {
    return scalar_face_y_raw(g, ph, f, phi, i, jface, 0.0, 0.0);
}

static void apply_bc_h_exact(const Grid *g, const double *vfield, double *h) {
    const int nx=g->Nx, ny=g->Ny;
    for (int j=1; j<=ny; ++j) {
        h[IDX(0,    j,ny)] = h[IDX(1, j,ny)];
        h[IDX(nx+1, j,ny)] = h[IDX(nx,j,ny)];
    }
    for (int i=1; i<=nx; ++i) {
        int slot = slot_cell(g, i);
        if (slot) {
            h[IDX(i,0,ny)]    = -h[IDX(i,1,ny)];
            h[IDX(i,ny+1,ny)] = -h[IDX(i,ny,ny)];
        } else {
            h[IDX(i,0,ny)]    = h[IDX(i,1,ny)];
            h[IDX(i,ny+1,ny)] = h[IDX(i,ny,ny)];
        }
    }
    (void)vfield;
}

static MAYBE_UNUSED void build_h_rhs_openfoam_like(const Grid *g, const Phys *ph, const Fields *f,
                                      const double *phi, double *rhs, double dt) {
    const int nx=g->Nx, ny=g->Ny; const double dx=g->dx, dy=g->dy;
    for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
        double Fe=u_face_x(g,f,i,j), Fw=u_face_x(g,f,i-1,j), Fn=v_face_y(g,ph,f,i,j), Fs=v_face_y(g,ph,f,i,j-1);
        double phiE=scalar_face_x_h(g,f,phi,i,j);
        double phiW=scalar_face_x_h(g,f,phi,i-1,j);
        double phiN=scalar_face_y_h(g,ph,f,phi,i,j);
        double phiS=scalar_face_y_h(g,ph,f,phi,i,j-1);
        rhs[IDX(i,j,ny)] = phi[IDX(i,j,ny)] - dt*((Fe*phiE - Fw*phiW)/dx + (Fn*phiN - Fs*phiS)/dy);
    }
}

static MAYBE_UNUSED void solve_h_helmholtz(const Grid *g, const Phys *ph, const Fields *f, double *phi, const double *rhs,
                              double dt, double diff, int iters) {
    (void)ph;
    const int nx=g->Nx, ny=g->Ny; const double ax=dt*diff/(g->dx*g->dx), ay=dt*diff/(g->dy*g->dy); const double ap=1.0+2.0*ax+2.0*ay;
    for (int it=0; it<iters; ++it) {
        apply_bc_h_exact(g, f->v, phi);
        for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
            double val = (rhs[IDX(i,j,ny)] + ax*(phi[IDX(i+1,j,ny)] + phi[IDX(i-1,j,ny)]) + ay*(phi[IDX(i,j+1,ny)] + phi[IDX(i,j-1,ny)]))/ap;
            phi[IDX(i,j,ny)] = MAX(val, 0.0);
        }
    }
    apply_bc_h_exact(g, f->v, phi);
}


static void cycle_history_prepare(CycleHistory *h, int capacity) {
    if (h->capacity >= capacity && h->absResidual && h->relResidual) return;
    free(h->absResidual);
    free(h->relResidual);
    h->absResidual = (double*)calloc((size_t)capacity, sizeof(double));
    h->relResidual = (double*)calloc((size_t)capacity, sizeof(double));
    if (!h->absResidual || !h->relResidual) die("allocate pressure cycle history");
    h->capacity = capacity;
    h->count = 0;
}

static void pressure_step_history_append(PressureStepHistory *h,
                                         const PressureStepRecord *rec) {
    if (h->count == h->capacity) {
        size_t newCapacity = h->capacity ? 2*h->capacity : 4096;
        PressureStepRecord *newData = (PressureStepRecord*)realloc(
            h->data, newCapacity*sizeof(PressureStepRecord));
        if (!newData) die("grow pressure step history");
        h->data = newData;
        h->capacity = newCapacity;
    }
    h->data[h->count++] = *rec;
}

static inline double gradp_x_cell(const Grid *g, const Fields *f, int i, int j) {
    const int ny=g->Ny, nx=g->Nx;
    if (i==1)  return (f->p[IDX(2,j,ny)]    - f->p[IDX(1,j,ny)]) / g->dx;
    if (i==nx) return (f->p[IDX(nx,j,ny)]   - f->p[IDX(nx-1,j,ny)]) / g->dx;
    return (f->p[IDX(i+1,j,ny)] - f->p[IDX(i-1,j,ny)]) / (2.0*g->dx);
}
static inline double gradp_y_cell(const Grid *g, const Fields *f, int i, int j) {
    const int ny=g->Ny;
    if (j==1)  return (f->p[IDX(i,2,ny)]    - f->p[IDX(i,1,ny)]) / g->dy;
    if (j==ny) return (f->p[IDX(i,ny,ny)]   - f->p[IDX(i,ny-1,ny)]) / g->dy;
    return (f->p[IDX(i,j+1,ny)] - f->p[IDX(i,j-1,ny)]) / (2.0*g->dy);
}

static void compute_face_flux_rc(const Grid *g, const Phys *ph, Fields *f, double dt, int usePredicted) {
    (void)ph;
    const int nx=g->Nx, ny=g->Ny;
    double *uc = usePredicted ? f->us : f->u;
    double *vc = usePredicted ? f->vs : f->v;
    const int applyRC = usePredicted ? 0 : 1;
    #pragma omp parallel for schedule(static)
    for (int j=1; j<=ny; ++j) {
        f->ufx[IDX(0,j,ny)] = uc[IDX(1,j,ny)];
        f->mfx[IDX(0,j,ny)] = rho_face_x(g, f, 0, j) * f->ufx[IDX(0,j,ny)];
        f->ufx[IDX(nx,j,ny)] = uc[IDX(nx,j,ny)];
        f->mfx[IDX(nx,j,ny)] = rho_face_x(g, f, nx, j) * f->ufx[IDX(nx,j,ny)];
        for (int i=1; i<nx; ++i) {
            double uf = 0.5*(uc[IDX(i,j,ny)] + uc[IDX(i+1,j,ny)]);
            if (applyRC) {
                double rhoFace = MAX(rho_face_x(g, f, i, j), 1.0e-12);
                double dpface = (f->p[IDX(i+1,j,ny)] - f->p[IDX(i,j,ny)]) / g->dx;
                double gpbar  = 0.5*(gradp_x_cell(g,f,i,j) + gradp_x_cell(g,f,i+1,j));
                uf -= (dt/rhoFace)*(dpface - gpbar);
            }
            f->ufx[IDX(i,j,ny)] = uf;
            f->mfx[IDX(i,j,ny)] = rho_face_x(g, f, i, j) * uf;
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i=1; i<=nx; ++i) {
        f->vfy[IDX(i,0,ny)] = slot_cell(g,i) ? ph->vFuel : 0.0;
        f->mfy[IDX(i,0,ny)] = rho_face_y(g, f, i, 0) * f->vfy[IDX(i,0,ny)];
        f->vfy[IDX(i,ny,ny)] = slot_cell(g,i) ? ph->vOx : 0.0;
        f->mfy[IDX(i,ny,ny)] = rho_face_y(g, f, i, ny) * f->vfy[IDX(i,ny,ny)];
        for (int j=1; j<ny; ++j) {
            double vf = 0.5*(vc[IDX(i,j,ny)] + vc[IDX(i,j+1,ny)]);
            if (applyRC) {
                double rhoFace = MAX(rho_face_y(g, f, i, j), 1.0e-12);
                double dpface = (f->p[IDX(i,j+1,ny)] - f->p[IDX(i,j,ny)]) / g->dy;
                double gpbar  = 0.5*(gradp_y_cell(g,f,i,j) + gradp_y_cell(g,f,i,j+1));
                vf -= (dt/rhoFace)*(dpface - gpbar);
            }
            f->vfy[IDX(i,j,ny)] = vf;
            f->mfy[IDX(i,j,ny)] = rho_face_y(g, f, i, j) * vf;
        }
    }
}

static double u_face_x(const Grid *g, const Fields *f, int iface, int j) {
    (void)g;
    return f->ufx[IDX(iface,j,g->Ny)];
}
static double v_face_y(const Grid *g, const Phys *ph, const Fields *f, int i, int jface) {
    (void)g; (void)ph;
    return f->vfy[IDX(i,jface,g->Ny)];
}
static double rho_face_x(const Grid *g, const Fields *f, int iface, int j) {
    if (iface <= 0) return f->rho[IDX(1,j,g->Ny)];
    if (iface >= g->Nx) return f->rho[IDX(g->Nx,j,g->Ny)];
    return 0.5*(f->rho[IDX(iface,j,g->Ny)] + f->rho[IDX(iface+1,j,g->Ny)]);
}
static double rho_face_y(const Grid *g, const Fields *f, int i, int jface) {
    if (jface <= 0) return f->rho[IDX(i,1,g->Ny)];
    if (jface >= g->Ny) return f->rho[IDX(i,g->Ny,g->Ny)];
    return 0.5*(f->rho[IDX(i,jface,g->Ny)] + f->rho[IDX(i,jface+1,g->Ny)]);
}
static MAYBE_UNUSED double v_face_x(const Grid *g, const Fields *f, int iface, int j) {
    const int ny = g->Ny, nx = g->Nx;
    /* Same narrow V64C patch for tangential velocity transport at the open side:
       use the internal boundary-cell value at the face instead of clipping to 0. */
    if (iface == 0)  return f->v[IDX(1, j, ny)];
    if (iface == nx) return f->v[IDX(nx,j, ny)];
    return 0.5*(f->v[IDX(iface,j,ny)] + f->v[IDX(iface+1,j,ny)]);
}
static MAYBE_UNUSED double u_face_y(const Grid *g, const Fields *f, int i, int jface) {
    const int ny = g->Ny;
    if (jface == 0 || jface == ny) return 0.0;
    return 0.5*(f->u[IDX(i,jface,ny)] + f->u[IDX(i,jface+1,ny)]);
}

static void momentum_predictor(const Grid *g, const Phys *ph, Fields *f, double dt, double relaxU) {
    const int nx = g->Nx, ny = g->Ny;
    const double dx = g->dx, dy = g->dy;
    const double dx2 = dx*dx, dy2 = dy*dy;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
        double uij=f->u[IDX(i,j,ny)], vij=f->v[IDX(i,j,ny)];
        double rhoi = MAX(f->rho[IDX(i,j,ny)], 1.0e-12);

        double ue=scalar_face_x_raw(g,f,f->u,i,j,0.0),   uw=scalar_face_x_raw(g,f,f->u,i-1,j,0.0);
        double un=scalar_face_y_raw(g,ph,f,f->u,i,j,0.0,0.0), us=scalar_face_y_raw(g,ph,f,f->u,i,j-1,0.0,0.0);
        double ve=scalar_face_x_raw(g,f,f->v,i,j,0.0),   vw=scalar_face_x_raw(g,f,f->v,i-1,j,0.0);
        double vn=scalar_face_y_raw(g,ph,f,f->v,i,j,0.0,0.0), vs=scalar_face_y_raw(g,ph,f,f->v,i,j-1,0.0,0.0);

        double mE=f->mfx[IDX(i,  j,ny)], mW=f->mfx[IDX(i-1,j,ny)];
        double mN=f->mfy[IDX(i,j,  ny)], mS=f->mfy[IDX(i,j-1,ny)];

        double du_adv = -((mE*ue - mW*uw)/dx + (mN*un - mS*us)/dy) / rhoi;
        double dv_adv = -((mE*ve - mW*vw)/dx + (mN*vn - mS*vs)/dy) / rhoi;

        double muE = face_avg_x_coeff(g, f->muMix, i,   j);
        double muW = face_avg_x_coeff(g, f->muMix, i-1, j);
        double muN = face_avg_y_coeff(g, f->muMix, i,   j);
        double muS = face_avg_y_coeff(g, f->muMix, i, j-1);
        double du_diff = ((muE*(f->u[IDX(i+1,j,ny)] - uij) - muW*(uij - f->u[IDX(i-1,j,ny)]))/dx2
                        + (muN*(f->u[IDX(i,j+1,ny)] - uij) - muS*(uij - f->u[IDX(i,j-1,ny)]))/dy2) / rhoi;
        double dv_diff = ((muE*(f->v[IDX(i+1,j,ny)] - vij) - muW*(vij - f->v[IDX(i-1,j,ny)]))/dx2
                        + (muN*(f->v[IDX(i,j+1,ny)] - vij) - muS*(vij - f->v[IDX(i,j-1,ny)]))/dy2) / rhoi;

        /* Non-incremental projection: pressure is intentionally excluded here and
           applied once, after the Poisson solve, in correct_velocity(). */
        f->us[IDX(i,j,ny)] = relaxU*(uij + dt*(du_adv + du_diff)) + (1.0-relaxU)*uij;
        f->vs[IDX(i,j,ny)] = relaxU*(vij + dt*(dv_adv + dv_diff)) + (1.0-relaxU)*vij;
    }
}

static void build_pressure_rhs(const Grid *g, const Phys *ph, Fields *f, double dt) {
    const int nx = g->Nx, ny = g->Ny;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
        double mE = f->mfx[IDX(i,  j,ny)];
        double mW = f->mfx[IDX(i-1,j,ny)];
        double mN = f->mfy[IDX(i,j,  ny)];
        double mS = f->mfy[IDX(i,j-1,ny)];
        double divm = (mE - mW)/g->dx + (mN - mS)/g->dy;
        double drhoDt = ph->variableRhoOn ? (f->rho[IDX(i,j,ny)] - f->rhoOld[IDX(i,j,ny)]) / MAX(dt,1.0e-20) : 0.0;
        f->rhs[IDX(i,j,ny)] = (divm + drhoDt) / MAX(dt,1.0e-20);
    }
}

static inline double face_avg_x_coeff(const Grid *g, const double *a, int i, int j) {
    if (i <= 0) return a[IDX(1,j,g->Ny)];
    if (i >= g->Nx) return a[IDX(g->Nx,j,g->Ny)];
    return 0.5*(a[IDX(i,j,g->Ny)] + a[IDX(i+1,j,g->Ny)]);
}

static inline double face_avg_y_coeff(const Grid *g, const double *a, int i, int j) {
    if (j <= 0) return a[IDX(i,1,g->Ny)];
    if (j >= g->Ny) return a[IDX(i,g->Ny,g->Ny)];
    return 0.5*(a[IDX(i,j,g->Ny)] + a[IDX(i,j+1,g->Ny)]);
}

static MAYBE_UNUSED double residual_norm_pressure_variable(const Grid *g, const Fields *f, const double *rhs) {
    const double dx2 = g->dx*g->dx, dy2 = g->dy*g->dy;
    double maxRes = 0.0;
    for (int i=1; i<=g->Nx; ++i) for (int j=1; j<=g->Ny; ++j) {
        double lap = (f->p[IDX(i+1,j,g->Ny)] - 2.0*f->p[IDX(i,j,g->Ny)] + f->p[IDX(i-1,j,g->Ny)])/dx2
                   + (f->p[IDX(i,j+1,g->Ny)] - 2.0*f->p[IDX(i,j,g->Ny)] + f->p[IDX(i,j-1,g->Ny)])/dy2;
        double r = fabs(lap - rhs[IDX(i,j,g->Ny)]);
        if (r > maxRes) maxRes = r;
    }
    return maxRes;
}

static double max_abs_interior(const Grid *g, const double *a) {
    double m = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:m) schedule(static)
    for (int i=1; i<=g->Nx; ++i) for (int j=1; j<=g->Ny; ++j) {
        double v = fabs(a[IDX(i,j,g->Ny)]);
        if (isfinite(v) && v > m) m = v;
    }
    return m;
}

static double residual_norm_scalar_variable(const Grid *g, const Fields *f, const double *phi, const double *rhs,
                                            const double *diffField, double dt) {
    const double dx2 = g->dx*g->dx, dy2 = g->dy*g->dy;
    double maxRes = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:maxRes) schedule(static)
    for (int i=1; i<=g->Nx; ++i) for (int j=1; j<=g->Ny; ++j) {
        double rhoP = MAX(f->rho[IDX(i,j,g->Ny)], 1.0e-12);
        double gE = rho_face_x(g, f, i,   j) * face_avg_x_coeff(g, diffField, i,   j) / dx2;
        double gW = rho_face_x(g, f, i-1, j) * face_avg_x_coeff(g, diffField, i-1, j) / dx2;
        double gN = rho_face_y(g, f, i,   j) * face_avg_y_coeff(g, diffField, i,   j) / dy2;
        double gS = rho_face_y(g, f, i, j-1) * face_avg_y_coeff(g, diffField, i, j-1) / dy2;
        double ap = rhoP + dt*(gE + gW + gN + gS);
        double Aphi = ap*phi[IDX(i,j,g->Ny)]
                    - dt*(gE*phi[IDX(i+1,j,g->Ny)] + gW*phi[IDX(i-1,j,g->Ny)]
                        + gN*phi[IDX(i,j+1,g->Ny)] + gS*phi[IDX(i,j-1,g->Ny)]);
        double r = fabs(Aphi - rhs[IDX(i,j,g->Ny)]);
        if (r > maxRes) maxRes = r;
    }
    return maxRes;
}

#ifdef RMT_ANISO
#include "rmta_impl.h"      /* RMTA: geometry-aware hierarchy, same interface */
#else
#include "rmt_impl.h"
#endif


/* ---------------- pressure-system dump hook (pbench) ---------------- */
static long g_pdumpCall = 0;
static double g_pdumpTime = -1.0;      /* LOWMACH: simulated time at the end of the current step */
static int pdump_time_wanted(void) {   /* PDUMP_TIMES=t1,t2,... : dump the first step reaching each t_k */
    static double tl[256]; static int nt = -1, k = 0;
    if (nt < 0) {
        nt = 0; const char *list = getenv("PDUMP_TIMES");
        if (list && *list) { char buf[8192]; snprintf(buf, sizeof(buf), "%s", list);
            for (char *tok = strtok(buf, ","); tok && nt < 256; tok = strtok(NULL, ",")) tl[nt++] = atof(tok); }
    }
    int hit = 0;
    while (k < nt && g_pdumpTime >= tl[k] - 1e-12) { hit = 1; ++k; }
    return hit;
}
static int pdump_wanted(long step) {
    if (pdump_time_wanted()) return 1;
    const char *list = getenv("PDUMP_STEPS");
    if (!list || !*list) return 0;
    char buf[8192];
    snprintf(buf, sizeof(buf), "%s", list);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        if (atol(tok) == step) return 1;
    }
    return 0;
}
static void pdump_defect(const Grid *g, const Fields *f, double *out) {
    const int nx=g->Nx, ny=g->Ny;
    const double ax=1.0/(g->dx*g->dx), ay=1.0/(g->dy*g->dy), ap=2.0*ax+2.0*ay;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) {
        const double Ap = ap*f->p[IDX(i,j,ny)]
                        - ax*(f->p[IDX(i+1,j,ny)] + f->p[IDX(i-1,j,ny)])
                        - ay*(f->p[IDX(i,j+1,ny)] + f->p[IDX(i,j-1,ny)]);
        out[(size_t)(i-1)*ny + (j-1)] = -f->rhs[IDX(i,j,ny)] - Ap;
    }
}
static void pdump_maybe(const Grid *g, const Phys *ph, Fields *f) {
    ++g_pdumpCall;
    if (!pdump_wanted(g_pdumpCall)) return;
    const char *dir = getenv("PDUMP_DIR"); if (!dir || !*dir) dir = ".";
    const int nx=g->Nx, ny=g->Ny;
    const size_t n=(size_t)nx*ny, ntot=(size_t)(nx+2)*(ny+2);
    double *rw=malloc(n*sizeof(double)), *rc=malloc(n*sizeof(double)), *pk=malloc(ntot*sizeof(double));
    double *rp=malloc(n*sizeof(double));   /* defect at the uniform guess p = p0 */
    if(!rw||!rc||!pk||!rp) die("pdump alloc");
    apply_bc_velocity_pressure(g, ph, f);
    pdump_defect(g, f, rw);
    memcpy(pk, f->p, ntot*sizeof(double));
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) f->p[IDX(i,j,ny)] = 0.0;
    apply_bc_velocity_pressure(g, ph, f);
    pdump_defect(g, f, rc);
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) f->p[IDX(i,j,ny)] = ph->p0;
    apply_bc_velocity_pressure(g, ph, f);
    pdump_defect(g, f, rp);
    memcpy(f->p, pk, ntot*sizeof(double));
    apply_bc_velocity_pressure(g, ph, f);
    const double rhsNorm = max_abs_interior(g, f->rhs);
    char path[PATH_MAX];
    snprintf(path, sizeof(path), "%s/psys_%dx%d_step%07ld.bin", dir, nx, ny, g_pdumpCall);
    FILE *fp=fopen(path,"wb"); if(!fp) die("pdump open");
    const char magic[8]="PSYS2\0\0";
    int hdr[3]={nx,ny,(int)g_pdumpCall};
    double dh[3]={g->dx,g->dy,rhsNorm};
    fwrite(magic,1,8,fp); fwrite(hdr,sizeof(int),3,fp); fwrite(dh,sizeof(double),3,fp);
    fwrite(rw,sizeof(double),n,fp); fwrite(rc,sizeof(double),n,fp); fwrite(rp,sizeof(double),n,fp);
    fclose(fp);
    fprintf(stderr,"PDUMP wrote %s t=%.9g\n",path,g_pdumpTime);
    free(rw); free(rc); free(pk); free(rp);
    const char *ex=getenv("PDUMP_EXIT_AFTER");
    if (ex && atol(ex)>0 && g_pdumpCall>=atol(ex)) { fprintf(stderr,"PDUMP exit after step %ld\n",g_pdumpCall); exit(0); }
}
/* -------------------------------------------------------------------- */

static PressureStats solve_pressure_poisson(const Grid *g, const Phys *ph, Fields *f,
                                             const Controls *c,
                                             CycleHistory *history) {
    PressureStats stats = {0, 0.0, 0.0, 0.0, 0.0, 1.0};
    const int nx=g->Nx, ny=g->Ny;
    const double ax=1.0/(g->dx*g->dx), ay=1.0/(g->dy*g->dy);
    const double ap=2.0*ax+2.0*ay;
    RMTV2Workspace *ws=rmt_v2_workspace(g);
    double *corr=ws->corr, *defect=ws->res;

    pdump_maybe(g, ph, f);
    cycle_history_prepare(history, c->poissonIters + 1);
    history->count = 0;
    apply_bc_velocity_pressure(g, ph, f);

    stats.rhsNorm = max_abs_interior(g, f->rhs);
    stats.initialResidual = residual_norm_pressure_variable(g, f, f->rhs);
    stats.absResidual = stats.initialResidual;
    stats.relResidual = stats.absResidual / MAX(stats.rhsNorm, 1.0e-30);
    history->absResidual[0] = stats.absResidual;
    history->relResidual[0] = stats.relResidual;
    history->count = 1;

    if (stats.absResidual <= c->pressureAbsTol || stats.relResidual <= c->pressureRelTol)
        return stats;

    if (fabs(c->mgOmega-1.0) > 1.0e-14)
        die("RMT requires the full Martynenko correction (rmtOmega=1)");

    RMTV2Config cfg;
    cfg.requestedLevels = MAX(0, c->mgPreSmooth);
    cfg.smoothSweeps = MAX(1, c->mgPostSmooth);
    cfg.nCoarsest = 5;

    for (int cyc=0; cyc<c->poissonIters; ++cyc) {
        apply_bc_velocity_pressure(g, ph, f);
        const double oldRes = residual_norm_pressure_variable(g, f, f->rhs);

        /* A=-L and b=-rhs, hence defect b-Ap = Lp-rhs. */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
            const double Ap = ap*f->p[IDX(i,j,ny)]
                            - ax*(f->p[IDX(i+1,j,ny)] + f->p[IDX(i-1,j,ny)])
                            - ay*(f->p[IDX(i,j+1,ny)] + f->p[IDX(i,j-1,ny)]);
            defect[IDX(i,j,ny)] = -f->rhs[IDX(i,j,ny)] - Ap;
        }

        RMTV2Diag d = {0,0,0,0,0};
        rmt_v2_build_correction(g, defect, corr, &cfg, &d);

        /* Martynenko sawtooth update: transfer is index remapping; add the
         * computed correction in full.  RMT does not require monotonic
         * residual decrease at every multigrid iteration. */
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j)
            f->p[IDX(i,j,ny)] += corr[IDX(i,j,ny)];

        apply_bc_velocity_pressure(g, ph, f);
        const double newRes = residual_norm_pressure_variable(g, f, f->rhs);
        if (!isfinite(newRes))
            die("RMT produced a non-finite pressure residual");

        const double ratio = newRes/MAX(oldRes,1.0e-300);
        g_rmtV2LastCycleRatio = ratio;
        g_rmtV2MaxCycleRatio = MAX(g_rmtV2MaxCycleRatio,ratio);
        if (newRes > oldRes) ++g_rmtV2NonmonotoneCycles;

        stats.cycles = cyc + 1;
        stats.absResidual = newRes;
        stats.relResidual = stats.absResidual / MAX(stats.rhsNorm, 1.0e-30);
        history->absResidual[history->count] = stats.absResidual;
        history->relResidual[history->count] = stats.relResidual;
        ++history->count;

        if (stats.absResidual <= c->pressureAbsTol || stats.relResidual <= c->pressureRelTol)
            break;
    }

    if (stats.absResidual > c->pressureAbsTol && stats.relResidual > c->pressureRelTol) {
        fprintf(stderr,
                "RMT_PRESSURE_NOT_CONVERGED cycles=%d absResidual=%.17g relResidual=%.17g absTol=%.17g relTol=%.17g\n",
                stats.cycles, stats.absResidual, stats.relResidual,
                c->pressureAbsTol, c->pressureRelTol);
        die("RMT pressure solve hit max cycles without meeting the tolerance");
    }

    if (stats.cycles > 0 && stats.initialResidual > 0.0 && stats.absResidual > 0.0)
        stats.convergenceFactor =
            pow(stats.absResidual/stats.initialResidual, 1.0/(double)stats.cycles);
    else
        stats.convergenceFactor = 0.0;

    return stats;
}

static void correct_velocity_and_face_flux(const Grid *g, const Phys *ph, Fields *f, double dt) {
    const int nx=g->Nx, ny=g->Ny;
    (void)ph;

    /* Cell-centred velocity correction. */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
        double dpdx;
        if (i == 1) dpdx = (f->p[IDX(i+1,j,ny)] - f->p[IDX(i,j,ny)]) / g->dx;
        else if (i == nx) dpdx = (f->p[IDX(i,j,ny)] - f->p[IDX(i-1,j,ny)]) / g->dx;
        else dpdx = (f->p[IDX(i+1,j,ny)] - f->p[IDX(i-1,j,ny)])/(2.0*g->dx);
        double dpdy;
        if (j == 1) dpdy = (f->p[IDX(i,j+1,ny)] - f->p[IDX(i,j,ny)]) / g->dy;
        else if (j == ny) dpdy = (f->p[IDX(i,j,ny)] - f->p[IDX(i,j-1,ny)]) / g->dy;
        else dpdy = (f->p[IDX(i,j+1,ny)] - f->p[IDX(i,j-1,ny)])/(2.0*g->dy);
        double rhoi = MAX(f->rho[IDX(i,j,ny)],1.0e-12);
        f->u[IDX(i,j,ny)] = f->us[IDX(i,j,ny)] - dt*dpdx/rhoi;
        f->v[IDX(i,j,ny)] = f->vs[IDX(i,j,ny)] - dt*dpdy/rhoi;
    }

    /*
       Conservative face-mass-flux correction.  The pressure equation is

           laplacian(p) = [div(m*) + (rho^{n+1}-rho^n)/dt] / dt,

       so m^{n+1} = m* - dt grad(p).  Applying this correction directly to
       the stored face fluxes makes the discrete continuity residual use the
       same operator as the Poisson solve.
    */
    #pragma omp parallel for schedule(static)
    for (int j=1; j<=ny; ++j) {
        /* Side pressure patches are represented by ghost-cell values. */
        double gradLeft  = (f->p[IDX(1,j,ny)]    - f->p[IDX(0,j,ny)])    / g->dx;
        double gradRight = (f->p[IDX(nx+1,j,ny)] - f->p[IDX(nx,j,ny)]) / g->dx;
        f->mfx[IDX(0,j,ny)]  -= dt*gradLeft;
        f->mfx[IDX(nx,j,ny)] -= dt*gradRight;
        f->ufx[IDX(0,j,ny)]  = f->mfx[IDX(0,j,ny)]  / MAX(rho_face_x(g,f,0,j),1.0e-12);
        f->ufx[IDX(nx,j,ny)] = f->mfx[IDX(nx,j,ny)] / MAX(rho_face_x(g,f,nx,j),1.0e-12);
        for (int i=1; i<nx; ++i) {
            double gradFace = (f->p[IDX(i+1,j,ny)] - f->p[IDX(i,j,ny)]) / g->dx;
            f->mfx[IDX(i,j,ny)] -= dt*gradFace;
            f->ufx[IDX(i,j,ny)] = f->mfx[IDX(i,j,ny)] / MAX(rho_face_x(g,f,i,j),1.0e-12);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i=1; i<=nx; ++i) {
        /* Bottom/top are prescribed inlet or impermeable-wall mass fluxes; dp/dn=0. */
        f->vfy[IDX(i,0,ny)]  = f->mfy[IDX(i,0,ny)]  / MAX(rho_face_y(g,f,i,0),1.0e-12);
        f->vfy[IDX(i,ny,ny)] = f->mfy[IDX(i,ny,ny)] / MAX(rho_face_y(g,f,i,ny),1.0e-12);
        for (int j=1; j<ny; ++j) {
            double gradFace = (f->p[IDX(i,j+1,ny)] - f->p[IDX(i,j,ny)]) / g->dy;
            f->mfy[IDX(i,j,ny)] -= dt*gradFace;
            f->vfy[IDX(i,j,ny)] = f->mfy[IDX(i,j,ny)] / MAX(rho_face_y(g,f,i,j),1.0e-12);
        }
    }
}

static void build_scalar_rhs_openfoam_like(const Grid *g, const Phys *ph, const Fields *f,
                                           const double *phi, double *rhs, double dt,
                                           double bottomSlot, double topSlot, double outletInletValue) {
    const int nx=g->Nx, ny=g->Ny; const double dx=g->dx, dy=g->dy;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
        double mE=f->mfx[IDX(i,  j,ny)], mW=f->mfx[IDX(i-1,j,ny)];
        double mN=f->mfy[IDX(i,j,  ny)], mS=f->mfy[IDX(i,j-1,ny)];
        double phiE=scalar_face_x_species(g,f,phi,i,j,outletInletValue);
        double phiW=scalar_face_x_species(g,f,phi,i-1,j,outletInletValue);
        double phiN=scalar_face_y_species(g,ph,f,phi,i,j,bottomSlot,topSlot);
        double phiS=scalar_face_y_species(g,ph,f,phi,i,j-1,bottomSlot,topSlot);
        rhs[IDX(i,j,ny)] = f->rho[IDX(i,j,ny)]*phi[IDX(i,j,ny)]
                         - dt*((mE*phiE - mW*phiW)/dx + (mN*phiN - mS*phiS)/dy);
    }
}

static void solve_scalar_helmholtz(const Grid *g, const Phys *ph, const Fields *f, double *phi, const double *rhs,
                                   double dt, const double *diffField, int iters, int sweepsPerCycle,
                                   double lower, double upper,
                                   double bottomSlot, double topSlot, double outletInletValue, double relaxA) {
    const int nx=g->Nx, ny=g->Ny;
    const double dx2 = g->dx*g->dx, dy2 = g->dy*g->dy;
    sweepsPerCycle = MAX(1, sweepsPerCycle);
    (void)ph;

    /*
       Scalar and enthalpy equations: variable-coefficient red-black Gauss-Seidel.
       Only the pressure algorithm differs between the RMT and RMTA builds, so differences
       between the two are caused by the pressure solver alone.
    */
    for (int cyc=0; cyc<iters; ++cyc) {
        apply_bc_scalar_of(g, f, phi, bottomSlot, topSlot, outletInletValue);
        for (int sweep=0; sweep<sweepsPerCycle; ++sweep) {
            for (int color=0; color<2; ++color) {
                #pragma omp parallel for collapse(2) schedule(static)
                for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
                    if (((i + j) & 1) != color) continue;
                    double rhoP = MAX(f->rho[IDX(i,j,ny)], 1.0e-12);
                    double gE = rho_face_x(g, f, i,   j) * face_avg_x_coeff(g, diffField, i,   j) / dx2;
                    double gW = rho_face_x(g, f, i-1, j) * face_avg_x_coeff(g, diffField, i-1, j) / dx2;
                    double gN = rho_face_y(g, f, i,   j) * face_avg_y_coeff(g, diffField, i,   j) / dy2;
                    double gS = rho_face_y(g, f, i, j-1) * face_avg_y_coeff(g, diffField, i, j-1) / dy2;
                    double ap = rhoP + dt*(gE + gW + gN + gS);
                    double phiGs = (rhs[IDX(i,j,ny)] + dt*(gE*phi[IDX(i+1,j,ny)] + gW*phi[IDX(i-1,j,ny)]
                                                        + gN*phi[IDX(i,j+1,ny)] + gS*phi[IDX(i,j-1,ny)])) / MAX(ap, 1.0e-20);
                    double val = (1.0-relaxA)*phi[IDX(i,j,ny)] + relaxA*phiGs;
                    phi[IDX(i,j,ny)] = clamp(val, lower, upper);
                }
                apply_bc_scalar_of(g, f, phi, bottomSlot, topSlot, outletInletValue);
            }
        }
        if (residual_norm_scalar_variable(g, f, phi, rhs, diffField, dt) < 1.0e-8) break;
    }
}

static double arrhenius_k(const Phys *ph, double T) {
    T = MAX(T, 200.0);
    double logk = log(ph->A) + ph->beta*log(T) - ph->Ta/T;
    if (logk > 700.0) logk = 700.0;
    if (logk < -700.0) return 0.0;
    return exp(logk);
}

static double implicit_extent_be(const Phys *ph, double dts, double T, double CF0, double CO0) {
    double k = arrhenius_k(ph, T);
    double xiMax = MIN(0.5*CF0, CO0);
    if (xiMax <= 0.0 || k <= 0.0) return 0.0;
    double xi = MIN(dts*k*CF0*CF0*CO0, 0.5*xiMax);
    xi = clamp(xi, 0.0, xiMax*(1.0-1e-12));
    for (int it=0; it<12; ++it) {
        double a = MAX(CF0 - 2.0*xi, 0.0), b = MAX(CO0 - xi, 0.0);
        double F = xi - dts*k*a*a*b;
        if (fabs(F) < 1e-14*MAX(1.0, xiMax)) break;
        double dF = 1.0 + dts*k*(4.0*a*b + a*a); /* derivative of -a^2 b gives +(4ab+a^2) */
        double xnew = xi - F/dF;
        if (!(xnew >= 0.0 && xnew <= xiMax) || !isfinite(xnew)) xnew = 0.5*(xi + clamp(xnew,0.0,xiMax));
        xnew = clamp(xnew, 0.0, xiMax);
        if (fabs(xnew - xi) < 1e-14*MAX(1.0, xiMax)) { xi = xnew; break; }
        xi = xnew;
    }
    return clamp(xi, 0.0, xiMax);
}

static void chemistry_cell_step_implicit(const Phys *ph, double dtOuter, double pCell, double *YF, double *YO, double *YP, double *h) {
    double yF=clamp(*YF,0.0,1.0), yO=clamp(*YO,0.0,0.21), yP=clamp(*YP,0.0,1.0);
    double hloc = MAX(*h, 0.0);
    double yN2 = MAX(0.0, 1.0 - yF - yO - yP);
    /* Use outer-step lagged temperature to avoid runaway self-acceleration within the same fluid step. */
    const double tempLag = MAX(T_from_h_mix(ph, yF, yO, yP, yN2, hloc, ph->Tref + hloc/MAX(ph->cp,1.0)), 200.0);
    if (yF <= 1e-16 || yO <= 1e-16 || !ph->chemistryOn) { *YF=yF; *YO=yO; *YP=yP; *h=hloc; return; }

    double rhoInit = mixture_density(ph, yF, yO, yP, yN2, tempLag, pCell);
    double CFinit = rhoInit * yF / ph->W_F;
    double COinit = rhoInit * yO / ph->W_O;
    double xiMaxInit = MIN(0.5*CFinit, COinit);
    if (xiMaxInit <= 0.0) { *YF=yF; *YO=yO; *YP=yP; *h=hloc; return; }

    /* Cumulative chemistry budget over the whole outer step: reactedFracMax applies to the entire step,
       not each accepted chemistry substep. */
    double xiBudget = ph->reactedFracMax * xiMaxInit;
    double xiCum = 0.0;
    double tloc=0.0;
    double dts = MIN(ph->initialChemicalTimeStep, dtOuter);
    double qcoeff = reaction_heat_release_per_extent(ph, tempLag);
    int guard=0;
    while (tloc < dtOuter - 1e-15 && ++guard < 400) {
        if (tloc + dts > dtOuter) dts = dtOuter - tloc;
        yN2 = MAX(0.0, 1.0 - yF - yO - yP);
        double rhoLoc = mixture_density(ph, yF, yO, yP, yN2, tempLag, pCell);
        double cpLoc = MAX(mixture_cp(ph, yF, yO, yP, yN2, tempLag), 1.0);
        double CF0 = rhoLoc * yF / ph->W_F;
        double CO0 = rhoLoc * yO / ph->W_O;
        double xiMax = MIN(0.5*CF0, CO0);
        if (xiMax <= 0.0) break;
        double xi = implicit_extent_be(ph, dts, tempLag, CF0, CO0);
        double xiRemain = xiBudget - xiCum;
        if (xiRemain <= 1e-20) break;
        if (xi > xiRemain) xi = xiRemain;
        double dh = qcoeff * xi / MAX(rhoLoc, 1.0e-12);
        double dT = dh / cpLoc;
        if (dT > ph->dTchemMax && dts > 1e-14) { dts *= 0.25; continue; }
        yF = clamp(yF - (2.0*ph->W_F*xi)/MAX(rhoLoc,1.0e-12), 0.0, 1.0);
        yO = clamp(yO - (1.0*ph->W_O*xi)/MAX(rhoLoc,1.0e-12), 0.0, 0.21);
        yP = clamp(yP + (2.0*ph->W_P*xi)/MAX(rhoLoc,1.0e-12), 0.0, 1.0);
        hloc = MAX(0.0, hloc + dh);
        xiCum += xi;
        /* Do not renormalize all active species together; preserve YF/YO shapes and only limit product to available remainder. */
        {
            double ypMax = MAX(0.0, 1.0 - yF - yO);
            if (ypMax < 0.0) ypMax = 0.0;
            yP = clamp(yP, 0.0, ypMax);
        }
        tloc += dts;
        dts = MIN(ph->maxChemicalTimeStep, dts*2.0);
    }
    *YF=yF; *YO=yO; *YP=yP; *h=hloc;
}

static void chemistry_subcycle(const Grid *g, const Phys *ph, Fields *f, double dt) {
    if (!ph->chemistryOn) return;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1;i<=g->Nx;++i) for (int j=1;j<=g->Ny;++j) {
        chemistry_cell_step_implicit(ph, dt, f->p[IDX(i,j,g->Ny)], &f->YF[IDX(i,j,g->Ny)], &f->YO[IDX(i,j,g->Ny)], &f->YP[IDX(i,j,g->Ny)], &f->h[IDX(i,j,g->Ny)]);
        double yF = clamp(f->YF[IDX(i,j,g->Ny)], 0.0, 1.0);
        double yO = clamp(f->YO[IDX(i,j,g->Ny)], 0.0, 1.0);
        double yP = clamp(f->YP[IDX(i,j,g->Ny)], 0.0, 1.0);
        double yN2 = MAX(0.0, 1.0 - yF - yO - yP);
        f->T[IDX(i,j,g->Ny)] = T_from_h_mix(ph, yF, yO, yP, yN2, f->h[IDX(i,j,g->Ny)], f->T[IDX(i,j,g->Ny)]);
    }
}

static void update_inert(const Grid *g, Fields *f) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=1;i<=g->Nx;++i) for (int j=1;j<=g->Ny;++j) {
        f->YF[IDX(i,j,g->Ny)] = clamp(f->YF[IDX(i,j,g->Ny)], 0.0, 1.0);
        f->YO[IDX(i,j,g->Ny)] = clamp(f->YO[IDX(i,j,g->Ny)], 0.0, 0.21);

        /* Preserve fuel/oxidizer fields and only trim product to the available active-species remainder. */
        {
            double ypMax = MAX(0.0, 1.0 - f->YF[IDX(i,j,g->Ny)] - f->YO[IDX(i,j,g->Ny)]);
            f->YP[IDX(i,j,g->Ny)] = clamp(f->YP[IDX(i,j,g->Ny)], 0.0, ypMax);
        }

        double sum = f->YF[IDX(i,j,g->Ny)] + f->YO[IDX(i,j,g->Ny)] + f->YP[IDX(i,j,g->Ny)];
        f->YN2[IDX(i,j,g->Ny)] = MAX(0.0, 1.0-sum);
    }
}

static double compute_max_co(const Grid *g, const Fields *f, double dt) {
    double m=0.0;
    #pragma omp parallel for collapse(2) reduction(max:m) schedule(static)
    for (int i=1;i<=g->Nx;++i) for (int j=1;j<=g->Ny;++j) {
        double co = fabs(f->u[IDX(i,j,g->Ny)])*dt/g->dx + fabs(f->v[IDX(i,j,g->Ny)])*dt/g->dy;
        if (co>m) m=co;
    }
    return m;
}

static double compute_max_diffusion_number(const Grid *g, const Fields *f, double dt) {
    const double metric = 1.0/(g->dx*g->dx) + 1.0/(g->dy*g->dy);
    double m = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:m) schedule(static)
    for (int i=1;i<=g->Nx;++i) for (int j=1;j<=g->Ny;++j) {
        double di = MAX(f->nuMix[IDX(i,j,g->Ny)], 0.0) * dt * metric;
        if (isfinite(di) && di > m) m = di;
    }
    return m;
}

static MassStats compute_mass_stats(const Grid *g, const Phys *ph, const Fields *f, double dt) {
    double maxAbs = 0.0, sum2 = 0.0, integral = 0.0;
    long long n = 0;
    #pragma omp parallel for collapse(2) reduction(max:maxAbs) reduction(+:sum2,integral,n) schedule(static)
    for (int i=1; i<=g->Nx; ++i) for (int j=1; j<=g->Ny; ++j) {
        double divm = (f->mfx[IDX(i,j,g->Ny)] - f->mfx[IDX(i-1,j,g->Ny)])/g->dx
                    + (f->mfy[IDX(i,j,g->Ny)] - f->mfy[IDX(i,j-1,g->Ny)])/g->dy;
        double drhoDt = ph->variableRhoOn
                      ? (f->rho[IDX(i,j,g->Ny)] - f->rhoOld[IDX(i,j,g->Ny)]) / MAX(dt,1.0e-30)
                      : 0.0;
        double r = drhoDt + divm;
        double ar = fabs(r);
        if (ar > maxAbs) maxAbs = ar;
        sum2 += r*r;
        integral += r*g->dx*g->dy;
        ++n;
    }
    MassStats st = {maxAbs, (n > 0) ? sqrt(sum2/(double)n) : 0.0, integral};
    return st;
}

static double field_max(const Grid *g, const double *a) {
    double m=-1e300;
    #pragma omp parallel for collapse(2) reduction(max:m) schedule(static)
    for(int i=1;i<=g->Nx;++i) for(int j=1;j<=g->Ny;++j) {
        double v=a[IDX(i,j,g->Ny)];
        if (isfinite(v) && v>m) m=v;
    }
    return m;
}

static double field_min(const Grid *g, const double *a) {
    double m=1e300;
    #pragma omp parallel for collapse(2) reduction(min:m) schedule(static)
    for(int i=1;i<=g->Nx;++i) for(int j=1;j<=g->Ny;++j) {
        double v=a[IDX(i,j,g->Ny)];
        if (isfinite(v) && v<m) m=v;
    }
    return m;
}

static void make_time_tag(double time, char *buf, size_t n) { snprintf(buf,n,"t%.6f",time); for(size_t k=0; buf[k]; ++k) if(buf[k]=='.') buf[k]='p'; }


static double point_u_value(const Grid *g, const Phys *ph, const Fields *f, int ip, int jp) {
    const int nx=g->Nx, ny=g->Ny;
    (void)ph;
    double x = ip * g->dx;
    int onSlot = (x >= g->x1 - 1e-12 && x <= g->x2 + 1e-12);

    if (jp == 0 || jp == ny) {
        if (onSlot) return 0.0; /* inlet velocity is purely vertical on slot faces */
        return 0.0;             /* walls */
    }

    if (ip == 0)  return f->u[IDX(1,  jp, ny)];
    if (ip == nx) return f->u[IDX(nx, jp, ny)];

    double u00 = f->u[IDX(ip,   jp,   ny)];
    double u10 = f->u[IDX(ip+1, jp,   ny)];
    double u01 = f->u[IDX(ip,   jp+1, ny)];
    double u11 = f->u[IDX(ip+1, jp+1, ny)];
    return 0.25*(u00 + u10 + u01 + u11);
}

static double point_v_value(const Grid *g, const Phys *ph, const Fields *f, int ip, int jp) {
    const int nx=g->Nx, ny=g->Ny;
    double x = ip * g->dx;
    int onSlot = (x >= g->x1 - 1e-12 && x <= g->x2 + 1e-12);

    if (jp == 0)  return onSlot ? ph->vFuel : 0.0;
    if (jp == ny) return onSlot ? ph->vOx   : 0.0;

    if (ip == 0)  return f->v[IDX(1,  jp, ny)];
    if (ip == nx) return f->v[IDX(nx, jp, ny)];

    double v00 = f->v[IDX(ip,   jp,   ny)];
    double v10 = f->v[IDX(ip+1, jp,   ny)];
    double v01 = f->v[IDX(ip,   jp+1, ny)];
    double v11 = f->v[IDX(ip+1, jp+1, ny)];
    return 0.25*(v00 + v10 + v01 + v11);
}

static void pvd_open(PvdWriter *pvd, const char *outputDir) {
    if (ensure_dir_recursive(outputDir) != 0) die("create outputDir");
    path_join(pvd->path, sizeof(pvd->path), outputDir, "fields.pvd");
    pvd->fp = fopen(pvd->path, "w");
    if (!pvd->fp) die("open fields.pvd");
    fprintf(pvd->fp, "<?xml version=\"1.0\"?>\n");
    fprintf(pvd->fp, "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(pvd->fp, "  <Collection>\n");
    fflush(pvd->fp);
}

static void pvd_add(PvdWriter *pvd, double time, const char *relativeFile) {
    if (!pvd || !pvd->fp) return;
    fprintf(pvd->fp, "    <DataSet timestep=\"%.16e\" group=\"\" part=\"0\" file=\"%s\"/>\n", time, relativeFile);
    fflush(pvd->fp);
}

static void pvd_close(PvdWriter *pvd) {
    if (!pvd || !pvd->fp) return;
    fprintf(pvd->fp, "  </Collection>\n</VTKFile>\n");
    fclose(pvd->fp);
    pvd->fp = NULL;
}

static void write_vti(const Grid *g, const Phys *ph, const Fields *f, double time,
                      const char *outputDir, PvdWriter *pvd) {
    { const char *nv = getenv("VDC_NO_VTI"); if (nv && atoi(nv)) return; }   /* no VTI output */
    char vtkDir[PATH_MAX];
    path_join(vtkDir, sizeof(vtkDir), outputDir, "vtk");
    if (ensure_dir_recursive(vtkDir) != 0) die("mkdir vtk");
    char tag[64], fname[PATH_MAX], relname[256];
    make_time_tag(time,tag,sizeof(tag));
    snprintf(relname,sizeof(relname),"vtk/field_%s.vti",tag);
    path_join(fname, sizeof(fname), outputDir, relname);
    FILE *fp=fopen(fname,"w"); if(!fp) die("open vtk");
    fprintf(fp,"<?xml version=\"1.0\"?>\n");
    fprintf(fp,"<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">\n");
    fprintf(fp,"  <ImageData WholeExtent=\"0 %d 0 %d 0 0\" Origin=\"0 -0.01 0\" Spacing=\"%.16e %.16e 1\">\n",g->Nx,g->Ny,g->dx,g->dy);
    fprintf(fp,"    <FieldData>\n");
    fprintf(fp,"      <DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">%.16e</DataArray>\n",time);
    fprintf(fp,"    </FieldData>\n");
    fprintf(fp,"    <Piece Extent=\"0 %d 0 %d 0 0\">\n",g->Nx,g->Ny);
    fprintf(fp,"      <CellData Scalars=\"T\">\n");
    #define WRITE_SCALAR(name,arr) do{ fprintf(fp,"        <DataArray type=\"Float64\" Name=\"%s\" format=\"ascii\">\n",name); \
        for(int j=1;j<=g->Ny;++j){ for(int i=1;i<=g->Nx;++i) fprintf(fp,"%.16e ",(arr)[IDX(i,j,g->Ny)]); fprintf(fp,"\n"); } \
        fprintf(fp,"        </DataArray>\n"); }while(0)
    WRITE_SCALAR("p",f->p); WRITE_SCALAR("T",f->T); WRITE_SCALAR("h",f->h); WRITE_SCALAR("rho",f->rho); WRITE_SCALAR("mu",f->muMix); WRITE_SCALAR("cpMix",f->cpMix); WRITE_SCALAR("YF",f->YF); WRITE_SCALAR("YO",f->YO); WRITE_SCALAR("YP",f->YP); WRITE_SCALAR("YN2",f->YN2); WRITE_SCALAR("u",f->u); WRITE_SCALAR("v",f->v);
    #undef WRITE_SCALAR
    fprintf(fp,"        <DataArray type=\"Float64\" Name=\"U\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for(int j=1;j<=g->Ny;++j){
        for(int i=1;i<=g->Nx;++i) fprintf(fp,"%.16e %.16e 0 ", f->u[IDX(i,j,g->Ny)], f->v[IDX(i,j,g->Ny)]);
        fprintf(fp,"\n");
    }
    fprintf(fp,"        </DataArray>\n");
    fprintf(fp,"      </CellData>\n");
    fprintf(fp,"      <PointData Vectors=\"U_point\">\n");
    fprintf(fp,"        <DataArray type=\"Float64\" Name=\"U_point\" NumberOfComponents=\"3\" format=\"ascii\">\n");
    for(int jp=0;jp<=g->Ny;++jp){
        for(int ip=0;ip<=g->Nx;++ip) fprintf(fp,"%.16e %.16e 0 ", point_u_value(g, ph, f, ip, jp), point_v_value(g, ph, f, ip, jp));
        fprintf(fp,"\n");
    }
    fprintf(fp,"        </DataArray>\n");
    fprintf(fp,"        <DataArray type=\"Float64\" Name=\"u_point\" format=\"ascii\">\n");
    for(int jp=0;jp<=g->Ny;++jp){ for(int ip=0;ip<=g->Nx;++ip) fprintf(fp,"%.16e ", point_u_value(g, ph, f, ip, jp)); fprintf(fp,"\n"); }
    fprintf(fp,"        </DataArray>\n");
    fprintf(fp,"        <DataArray type=\"Float64\" Name=\"v_point\" format=\"ascii\">\n");
    for(int jp=0;jp<=g->Ny;++jp){ for(int ip=0;ip<=g->Nx;++ip) fprintf(fp,"%.16e ", point_v_value(g, ph, f, ip, jp)); fprintf(fp,"\n"); }
    fprintf(fp,"        </DataArray>\n");
    fprintf(fp,"        <DataArray type=\"Float64\" Name=\"Umag_point\" format=\"ascii\">\n");
    for(int jp=0;jp<=g->Ny;++jp){
        for(int ip=0;ip<=g->Nx;++ip){
            double up = point_u_value(g, ph, f, ip, jp);
            double vp = point_v_value(g, ph, f, ip, jp);
            fprintf(fp,"%.16e ", sqrt(up*up + vp*vp));
        }
        fprintf(fp,"\n");
    }
    fprintf(fp,"        </DataArray>\n");
    fprintf(fp,"      </PointData>\n");
    fprintf(fp,"    </Piece>\n  </ImageData>\n</VTKFile>\n");
    fclose(fp);
    pvd_add(pvd, time, relname);
}

static void write_centerline_csv(const Grid *g, const Fields *f, double time, const char *outputDir) {
    char csvDir[PATH_MAX];
    path_join(csvDir, sizeof(csvDir), outputDir, "csv");
    if (ensure_dir_recursive(csvDir) != 0) die("mkdir csv");
    char tag[64], fname[PATH_MAX], leaf[128];
    make_time_tag(time,tag,sizeof(tag));
    snprintf(leaf,sizeof(leaf),"centerline_%s.csv",tag);
    path_join(fname, sizeof(fname), csvDir, leaf);
    FILE *fp=fopen(fname,"w"); if(!fp) return; fprintf(fp,"y,T,YF,YO,YP,u,v,p\n"); int i=g->Nx/2;
    for(int j=1;j<=g->Ny;++j){ double y=-0.01 + (j-0.5)*g->dy; fprintf(fp,"%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e,%.16e\n", y,f->T[IDX(i,j,g->Ny)],f->YF[IDX(i,j,g->Ny)],f->YO[IDX(i,j,g->Ny)],f->YP[IDX(i,j,g->Ny)],f->u[IDX(i,j,g->Ny)],f->v[IDX(i,j,g->Ny)],f->p[IDX(i,j,g->Ny)]); }
    fclose(fp);
}


/* ------------------------------------------------------------------------------------------
   LOWMACH diagnostics (read-only; enabled with -lowMachDiag 1).
   Pressure equation (mass-flux projection, with -perfectGas 1):
       lap(p) = [ div(m*) + (rho^{n+1}-rho^n)/dt ] / dt ,   m^{n+1} = m* - dt grad(p)
   The operator stays the constant-coefficient Laplacian, so RMT/RMTA/MG are unchanged; the
   reaction reaches the pressure solver through rho(T,Y) = p0 W/(R T) in m* = rho u* and in the
   thermal-expansion source (rho^{n+1}-rho^n)/dt.  These routines only measure that coupling.
   ------------------------------------------------------------------------------------------ */
typedef struct {
    int step; double time, dt;
    double rhoMin, rhoMax;
    double rhsL2, divL2, expL2;      /* L2 norms of the RHS and of its two parts (both /dt) */
    double rhsMax, divMax, expMax;   /* max norms */
    double expIntegral;              /* int (rho^{n+1}-rho^n)/dt dV  [kg/(m s)] */
    double massOutNet;               /* sum over boundary faces of m.n  [kg/(m s)] */
    double contMax;                  /* max |drho/dt + div m^{n+1}| after correction */
    double dilatMax, dilatIntegral;  /* max |div u| and int div u dV = net volume outflow [m^2/s] */
    /* global reaction balance (2F + O -> 2P), per unit depth, from the chemistry sub-step */
    double omegaF, omegaO, omegaP;   /* int rho dY/dt|chem dV  [kg/(m s)] (F,O consumed > 0, P produced > 0) */
    double qChem;                    /* int rho dh/dt|chem dV  [W/m] = heat release rate */
    double volChem;                  /* volume created by the chemistry sub-step at fixed mass, p0 [m^2/s] */
    /* global energy and fuel balances: storage + boundary out-flux */
    double hOut, hDiffIn, hStore;    /* advective h out-flux, conductive loss into the slots, d/dt int rho h dV [W/m] */
    double fIn, fOut, fStore;        /* fuel mass in/out through boundaries (adv.+diff.), d/dt int rho YF dV [kg/(m s)] */
    double hLag, fLag;               /* int phi^{n+1} (rho^{n+1}-rho^n)/dt dV: density-lag term of the scalar equations */
    int cycles;
} LowMachRecord;

typedef struct { LowMachRecord *data; size_t count, capacity; } LowMachHistory;

static void lowmach_pre(const Grid *g, const Phys *ph, const Fields *f, double dt, LowMachRecord *r) {
    const int nx=g->Nx, ny=g->Ny; const double idt = 1.0/MAX(dt,1.0e-20);
    double rmin=1e300, rmax=-1e300, s2=0, d2=0, e2=0, smax=0, dmax=0, emax=0, eint=0;
    #pragma omp parallel for collapse(2) reduction(min:rmin) reduction(max:rmax,smax,dmax,emax) reduction(+:s2,d2,e2,eint) schedule(static)
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) {
        double divm = (f->mfx[IDX(i,j,ny)]-f->mfx[IDX(i-1,j,ny)])/g->dx + (f->mfy[IDX(i,j,ny)]-f->mfy[IDX(i,j-1,ny)])/g->dy;
        double ex = ph->variableRhoOn ? (f->rho[IDX(i,j,ny)]-f->rhoOld[IDX(i,j,ny)])*idt : 0.0;
        double a = divm*idt, b = ex*idt, t = a+b;
        s2+=t*t; d2+=a*a; e2+=b*b;
        if (fabs(t)>smax) smax=fabs(t);
        if (fabs(a)>dmax) dmax=fabs(a);
        if (fabs(b)>emax) emax=fabs(b);
        eint += ex*g->dx*g->dy;
        double rh=f->rho[IDX(i,j,ny)]; if(rh<rmin) rmin=rh; if(rh>rmax) rmax=rh;
    }
    double n=(double)nx*(double)ny;
    r->rhoMin=rmin; r->rhoMax=rmax;
    r->rhsL2=sqrt(s2/n); r->divL2=sqrt(d2/n); r->expL2=sqrt(e2/n);
    r->rhsMax=smax; r->divMax=dmax; r->expMax=emax; r->expIntegral=eint;
}

static inline double lowmach_dilatation(const Grid *g, const Fields *f, int i, int j) {
    const int ny=g->Ny;
    return (f->ufx[IDX(i,j,ny)]-f->ufx[IDX(i-1,j,ny)])/g->dx + (f->vfy[IDX(i,j,ny)]-f->vfy[IDX(i,j-1,ny)])/g->dy;
}

static void lowmach_post(const Grid *g, const Phys *ph, const Fields *f, double dt, LowMachRecord *r) {
    const int nx=g->Nx, ny=g->Ny;
    double dmax=0, dint=0, cmax=0;
    #pragma omp parallel for collapse(2) reduction(max:dmax,cmax) reduction(+:dint) schedule(static)
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) {
        double du = lowmach_dilatation(g,f,i,j);
        if (fabs(du)>dmax) dmax=fabs(du);
        dint += du*g->dx*g->dy;
        double divm = (f->mfx[IDX(i,j,ny)]-f->mfx[IDX(i-1,j,ny)])/g->dx + (f->mfy[IDX(i,j,ny)]-f->mfy[IDX(i,j-1,ny)])/g->dy;
        double ex = ph->variableRhoOn ? (f->rho[IDX(i,j,ny)]-f->rhoOld[IDX(i,j,ny)])/MAX(dt,1.0e-30) : 0.0;
        if (fabs(divm+ex)>cmax) cmax=fabs(divm+ex);
    }
    double mout=0;
    for (int j=1;j<=ny;++j) mout += (f->mfx[IDX(nx,j,ny)] - f->mfx[IDX(0,j,ny)])*g->dy;
    for (int i=1;i<=nx;++i) mout += (f->mfy[IDX(i,ny,ny)] - f->mfy[IDX(i,0,ny)])*g->dx;
    r->dilatMax=dmax; r->dilatIntegral=dint; r->contMax=cmax; r->massOutNet=mout;
}

static void lowmach_append(LowMachHistory *h, const LowMachRecord *r) {
    if (h->count==h->capacity) {
        size_t nc = h->capacity ? 2*h->capacity : 4096;
        LowMachRecord *d=(LowMachRecord*)realloc(h->data, nc*sizeof(LowMachRecord));
        if(!d) die("grow lowmach history");
        h->data=d; h->capacity=nc;
    }
    h->data[h->count++]=*r;
}

typedef struct { double *YF, *YO, *YP, *h; double Eprev, Fprev; int havePrev; } LowMachChemBuf;

static void lowmach_chem_before(const Grid *g, const Fields *f, LowMachChemBuf *b) {
    size_t n = (size_t)(g->Nx+2)*(size_t)(g->Ny+2);
    if (!b->YF) { b->YF=malloc(n*sizeof(double)); b->YO=malloc(n*sizeof(double)); b->YP=malloc(n*sizeof(double)); b->h=malloc(n*sizeof(double));
                  if(!b->YF||!b->YO||!b->YP||!b->h) die("lowmach chem buffers"); }
    memcpy(b->YF,f->YF,n*sizeof(double)); memcpy(b->YO,f->YO,n*sizeof(double));
    memcpy(b->YP,f->YP,n*sizeof(double)); memcpy(b->h,f->h,n*sizeof(double));
}

static double lowmach_eos(const Phys *ph, double yF, double yO, double yP, double h, double Tg) {
    yF=clamp(yF,0,1); yO=clamp(yO,0,1); yP=clamp(yP,0,1); double yN=MAX(0.0,1.0-yF-yO-yP);
    double T = T_from_h_mix(ph,yF,yO,yP,yN,h,Tg);
    return mixture_density(ph,yF,yO,yP,yN,MAX(T,200.0),ph->p0);
}

static void lowmach_chem_after(const Grid *g, const Phys *ph, const Fields *f, const LowMachChemBuf *b, double dt, LowMachRecord *r) {
    const int nx=g->Nx, ny=g->Ny; const double dV=g->dx*g->dy, idt=1.0/MAX(dt,1e-30);
    double wF=0,wO=0,wP=0,q=0,vol=0;
    #pragma omp parallel for collapse(2) reduction(+:wF,wO,wP,q,vol) schedule(static)
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) {
        size_t k=IDX(i,j,ny); double m=f->rho[k]*dV;
        wF += m*(b->YF[k]-f->YF[k]); wO += m*(b->YO[k]-f->YO[k]); wP += m*(f->YP[k]-b->YP[k]); q += m*(f->h[k]-b->h[k]);
        if (ph->variableRhoOn && fabs(f->YF[k]-b->YF[k])>0.0) {
            double rb=lowmach_eos(ph,b->YF[k],b->YO[k],b->YP[k],b->h[k],f->T[k]);
            double ra=lowmach_eos(ph,f->YF[k],f->YO[k],f->YP[k],f->h[k],f->T[k]);
            vol += m*(1.0/ra-1.0/rb);
        }
    }
    r->omegaF=wF*idt; r->omegaO=wO*idt; r->omegaP=wP*idt; r->qChem=q*idt; r->volChem=vol*idt;
}

/* Boundary out-fluxes of phi using the solver's own face values and boundary conditions:
   advective m_f*phi_f (scalar_face_*_species) + diffusive Gamma_f*(phi_P - phi_ghost)/dn, ghost from apply_bc_scalar_of
   (outlet: zeroGradient on outflow, Dirichlet inletValue on inflow; slots: Dirichlet; walls: zero flux). */
static void lowmach_bflux(const Grid *g, const Phys *ph, const Fields *f, const double *phi, const double *diff,
                          double bottomSlot, double topSlot, double outletVal, double *adv, double *dif) {
    const int nx=g->Nx, ny=g->Ny; const double dx=g->dx, dy=g->dy;
    double A=0, D=0;
    for (int j=1;j<=ny;++j) {
        double pL=phi[IDX(1,j,ny)], pR=phi[IDX(nx,j,ny)];
        A += (-f->mfx[IDX(0,j,ny)]*scalar_face_x_species(g,f,phi,0,j,outletVal)
              + f->mfx[IDX(nx,j,ny)]*scalar_face_x_species(g,f,phi,nx,j,outletVal))*dy;
        double gL=rho_face_x(g,f,0,j)*face_avg_x_coeff(g,diff,0,j), gR=rho_face_x(g,f,nx,j)*face_avg_x_coeff(g,diff,nx,j);
        double ghL = (f->u[IDX(1,j,ny)]  < 0.0) ? pL : 2.0*outletVal - pL;
        double ghR = (f->u[IDX(nx,j,ny)] > 0.0) ? pR : 2.0*outletVal - pR;
        D += (gL*(pL-ghL) + gR*(pR-ghR))/dx*dy;
    }
    for (int i=1;i<=nx;++i) {
        double pB=phi[IDX(i,1,ny)], pT=phi[IDX(i,ny,ny)];
        A += (-f->mfy[IDX(i,0,ny)]*scalar_face_y_species(g,ph,f,phi,i,0,bottomSlot,topSlot)
              + f->mfy[IDX(i,ny,ny)]*scalar_face_y_species(g,ph,f,phi,i,ny,bottomSlot,topSlot))*dx;
        if (slot_cell(g,i)) {
            double gB=rho_face_y(g,f,i,0)*face_avg_y_coeff(g,diff,i,0), gT=rho_face_y(g,f,i,ny)*face_avg_y_coeff(g,diff,i,ny);
            D += (gB*(pB-(2.0*bottomSlot-pB)) + gT*(pT-(2.0*topSlot-pT)))/dy*dx;
        }
    }
    *adv=A; *dif=D;
}

static void lowmach_global(const Grid *g, const Phys *ph, const Fields *f, double dt, LowMachChemBuf *b, LowMachRecord *r) {
    const int nx=g->Nx, ny=g->Ny; const double dV=g->dx*g->dy;
    double hA,hD,fA,fD,E=0,F=0,LH=0,LF=0; const double idt=1.0/MAX(dt,1e-30);
    lowmach_bflux(g,ph,f,f->h,f->alphaMix,0.0,0.0,0.0,&hA,&hD);
    lowmach_bflux(g,ph,f,f->YF,f->DMix,ph->YF_fuel,ph->YF_ox,0.0,&fA,&fD);
    #pragma omp parallel for collapse(2) reduction(+:E,F,LH,LF) schedule(static)
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) { size_t k=IDX(i,j,ny); E+=f->rho[k]*f->h[k]*dV; F+=f->rho[k]*f->YF[k]*dV;
        double dr=(f->rho[k]-f->rhoOld[k])*idt*dV; LH+=f->h[k]*dr; LF+=f->YF[k]*dr; }
    r->hLag=LH; r->fLag=LF;
    r->hStore = b->havePrev ? (E-b->Eprev)/MAX(dt,1e-30) : 0.0;
    r->fStore = b->havePrev ? (F-b->Fprev)/MAX(dt,1e-30) : 0.0;
    b->Eprev=E; b->Fprev=F; b->havePrev=1;
    r->hOut=hA; r->hDiffIn=hD;          /* net advective / diffusive out-flux of h [W/m] */
    r->fOut=fA+fD; r->fIn=0.0;          /* net (adv.+diff.) out-flux of fuel [kg/(m s)]; in-flux folded in (negative) */
}

static void write_lowmach_history(const Controls *c, const LowMachHistory *h) {
    char perfDir[PATH_MAX], path[PATH_MAX];
    path_join(perfDir, sizeof(perfDir), c->outputDir, "performance");
    if (ensure_dir_recursive(perfDir) != 0) die("create performance directory");
    path_join(path, sizeof(path), perfDir, "lowmach_step_history.csv");
    FILE *fp=fopen(path,"w"); if(!fp) die("open lowmach_step_history.csv");
    fprintf(fp,"step,time_s,dt_s,pressure_cycles,rho_min,rho_max,rhs_L2,divm_L2,expansion_L2,rhs_max,divm_max,expansion_max,"
               "expansion_integral,mass_out_net,continuity_max,dilatation_max,dilatation_integral,"
               "omega_F,omega_O,omega_P,q_chem,vol_chem,h_out,h_diff_slots,h_storage,fuel_in,fuel_out,fuel_storage,h_lag,fuel_lag\n");
    for (size_t k=0;k<h->count;++k){ const LowMachRecord *r=&h->data[k];
        fprintf(fp,"%d,%.10g,%.10g,%d,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,"
                   "%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g,%.10g\n",
            r->step,r->time,r->dt,r->cycles,r->rhoMin,r->rhoMax,r->rhsL2,r->divL2,r->expL2,r->rhsMax,r->divMax,r->expMax,
            r->expIntegral,r->massOutNet,r->contMax,r->dilatMax,r->dilatIntegral,
            r->omegaF,r->omegaO,r->omegaP,r->qChem,r->volChem,r->hOut,r->hDiffIn,r->hStore,r->fIn,r->fOut,r->fStore,r->hLag,r->fLag); }
    fclose(fp);
}

static void write_lowmach_centerline(const Grid *g, const Fields *f, double time, const char *outputDir) {
    char csvDir[PATH_MAX], tag[64], fname[PATH_MAX], leaf[128];
    path_join(csvDir, sizeof(csvDir), outputDir, "csv");
    if (ensure_dir_recursive(csvDir) != 0) die("mkdir csv");
    make_time_tag(time,tag,sizeof(tag));
    snprintf(leaf,sizeof(leaf),"lowmach_centerline_%s.csv",tag);
    path_join(fname, sizeof(fname), csvDir, leaf);
    FILE *fp=fopen(fname,"w"); if(!fp) return;
    fprintf(fp,"y,T,rho,v,divu,p,YF,YO,YP\n"); int i=g->Nx/2;
    for(int j=1;j<=g->Ny;++j){ double y=-0.01+(j-0.5)*g->dy;
        fprintf(fp,"%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",y,f->T[IDX(i,j,g->Ny)],f->rho[IDX(i,j,g->Ny)],
                f->v[IDX(i,j,g->Ny)],lowmach_dilatation(g,f,i,j),f->p[IDX(i,j,g->Ny)],f->YF[IDX(i,j,g->Ny)],f->YO[IDX(i,j,g->Ny)],f->YP[IDX(i,j,g->Ny)]); }
    fclose(fp);
}

static void write_pressure_step_history(const Controls *c, const PressureStepHistory *h) {
    char perfDir[PATH_MAX], path[PATH_MAX];
    path_join(perfDir, sizeof(perfDir), c->outputDir, "performance");
    if (ensure_dir_recursive(perfDir) != 0) die("create performance directory");
    path_join(path, sizeof(path), perfDir, "pressure_step_history.csv");
    FILE *fp = fopen(path, "w");
    if (!fp) die("open pressure_step_history.csv");
    fprintf(fp, "step,time_s,dt_s,rbgs_sweeps,initial_residual,final_residual,final_relative_residual,convergence_factor\n");
    for (size_t k=0; k<h->count; ++k) {
        const PressureStepRecord *r = &h->data[k];
        fprintf(fp, "%d,%.17g,%.17g,%d,%.17g,%.17g,%.17g,%.17g\n",
                r->step, r->time, r->dt, r->cycles,
                r->initialResidual, r->finalResidual,
                r->finalRelativeResidual, r->convergenceFactor);
    }
    fclose(fp);
}

static void write_final_cycle_history(const Controls *c, const CycleHistory *h) {
    char perfDir[PATH_MAX], path[PATH_MAX];
    path_join(perfDir, sizeof(perfDir), c->outputDir, "performance");
    if (ensure_dir_recursive(perfDir) != 0) die("create performance directory");
    path_join(path, sizeof(path), perfDir, "final_pressure_cycle_history.csv");
    FILE *fp = fopen(path, "w");
    if (!fp) die("open final_pressure_cycle_history.csv");
    fprintf(fp, "sweep,absolute_residual,relative_residual,residual_ratio_to_previous\n");
    for (int k=0; k<h->count; ++k) {
        double ratio = (k > 0 && h->absResidual[k-1] > 0.0)
                     ? h->absResidual[k]/h->absResidual[k-1] : 1.0;
        fprintf(fp, "%d,%.17g,%.17g,%.17g\n",
                k, h->absResidual[k], h->relResidual[k], ratio);
    }
    fclose(fp);
}

static void write_case_summary_csv(const Grid *g, const Phys *ph, const Controls *c,
                                   const Fields *f, double finalTime, int steps,
                                   double computeElapsed, double pressureElapsed,
                                   double outputElapsed, double totalWall,
                                   long long pressureSolves, long long totalPressureSweeps,
                                   int maxPressureSweeps, double meanConvergenceFactor,
                                   const PressureStats *lastPressure,
                                   const MassStats *mass) {
    char path[PATH_MAX];
    path_join(path, sizeof(path), c->outputDir, "summary.csv");
    FILE *fp = fopen(path, "w");
    if (!fp) die("open summary.csv");

    const double avgSweeps = pressureSolves > 0
                           ? (double)totalPressureSweeps/(double)pressureSolves : 0.0;
    const double rmtEquivalentFineSweeps = (g->Nx > 0 && g->Ny > 0)
        ? (double)g_rmtV2PointUpdatesTotal/((double)g->Nx*(double)g->Ny) : 0.0;
    const double rmtEquivalentFineSweepsPerSolve = pressureSolves > 0
        ? rmtEquivalentFineSweeps/(double)pressureSolves : 0.0;
    const double pressureFraction = computeElapsed > 0.0
                                  ? 100.0*pressureElapsed/computeElapsed : 0.0;
    const double Tmax = field_max(g, f->T), Tmin = field_min(g, f->T);
    const double pmax = field_max(g, f->p), pmin = field_min(g, f->p);
    const double umax = field_max(g, f->u), umin = field_min(g, f->u);
    const double vmax = field_max(g, f->v), vmin = field_min(g, f->v);
    const double YFmax = field_max(g, f->YF), YOmax = field_max(g, f->YO);
    const double YPmax = field_max(g, f->YP), YN2max = field_max(g, f->YN2);

    fprintf(fp,
        "case_id,solver,threads,Nx,Ny,cells,level0_Nx,level0_Ny,rmt_smoothing_sweeps,rmt_max_cycles,final_time_s,time_steps,"
        "vFuel_m_per_s,vOx_m_per_s,A,beta,Ta_K,HfF_J_per_kg,rho_kg_per_m3,mu_Pa_s,cp_J_per_kgK,Pr,Sc,"
        "compute_time_s,pressure_time_s,pressure_fraction_pct,output_time_s,total_wall_s,"
        "pressure_solves,total_rmt_cycles,avg_rmt_cycles_per_pressure_solve,max_rmt_cycles_per_pressure_solve,"
        "mean_pressure_convergence_factor,final_pressure_cycles,final_pressure_rhs_norm,"
        "final_pressure_initial_residual,final_pressure_abs_residual,final_pressure_rel_residual,"
        "final_pressure_convergence_factor,equivalent_fine_grid_smoothing_sweeps,equivalent_fine_grid_smoothing_sweeps_per_pressure_solve,"
        "mass_residual_max,mass_residual_l2,mass_residual_integral,Tmin_K,Tmax_K,pmin_Pa,pmax_Pa,"
        "umin_m_per_s,umax_m_per_s,vmin_m_per_s,vmax_m_per_s,YFmax,YOmax,YPmax,YN2max\n");

    fprintf(fp, "%s,RMT,%d,%d,%d,%lld,%d,%d,%.17g,%d,%.17g,%d",
            c->caseId, c->threads, g->Nx, g->Ny,
            (long long)g->Nx*(long long)g->Ny, g->Nx, g->Ny,
            (double)c->mgPostSmooth, c->poissonIters, finalTime, steps);
    fprintf(fp, ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g",
            ph->vFuel, ph->vOx, ph->A, ph->beta, ph->Ta, ph->Hf_F,
            ph->rho, ph->mu, ph->cp, ph->Pr, ph->Sc);
    fprintf(fp, ",%.17g,%.17g,%.17g,%.17g,%.17g",
            computeElapsed, pressureElapsed, pressureFraction, outputElapsed, totalWall);
    fprintf(fp, ",%lld,%lld,%.17g,%d,%.17g,%d,%.17g,%.17g,%.17g,%.17g,%.17g",
            pressureSolves, totalPressureSweeps, avgSweeps, maxPressureSweeps,
            meanConvergenceFactor, lastPressure->cycles, lastPressure->rhsNorm,
            lastPressure->initialResidual, lastPressure->absResidual,
            lastPressure->relResidual, lastPressure->convergenceFactor);
    fprintf(fp, ",%.17g,%.17g", rmtEquivalentFineSweeps, rmtEquivalentFineSweepsPerSolve);
    fprintf(fp, ",%.17g,%.17g,%.17g", mass->maxAbs, mass->l2, mass->integral);
    fprintf(fp, ",%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",
            Tmin, Tmax, pmin, pmax, umin, umax, vmin, vmax,
            YFmax, YOmax, YPmax, YN2max);
    fclose(fp);
}

static void timer_now(struct timespec *t) {
    if (clock_gettime(CLOCK_MONOTONIC, t) != 0) die("clock_gettime");
}

static double elapsed_seconds(const struct timespec *a, const struct timespec *b) {
    return (double)(b->tv_sec - a->tv_sec) + 1.0e-9*(double)(b->tv_nsec - a->tv_nsec);
}

/* ---------------- state checkpoints and restart (no effect on the physics) ---------------- */
#define VDC_NF 14
static double **vdc_arrays(Fields *f, double **a) {
    a[0]=f->p; a[1]=f->u; a[2]=f->v; a[3]=f->T; a[4]=f->h; a[5]=f->YF; a[6]=f->YO; a[7]=f->YP; a[8]=f->YN2;
    a[9]=f->rho; a[10]=f->ufx; a[11]=f->vfy; a[12]=f->mfx; a[13]=f->mfy;
    return a;
}
/* CKPT_TIMES=t1,t2,... : write the state after the first step that reaches each t_k (file name uses t_k) */
static void vdc_checkpoint_maybe(const Grid *g, Fields *f, double time, double dtNext, int step) {
    static double tl[256]; static int nt=-1, k=0;
    if (nt < 0) {
        nt = 0; const char *list = getenv("CKPT_TIMES");
        if (list && *list) { char buf[8192]; snprintf(buf, sizeof(buf), "%s", list);
            for (char *tok=strtok(buf, ","); tok && nt<256; tok=strtok(NULL, ",")) tl[nt++] = atof(tok); }
    }
    while (k < nt && time >= tl[k] - 1e-12) {
        const char *dir = getenv("CKPT_DIR"); if (!dir || !*dir) dir = ".";
        char path[PATH_MAX];
        snprintf(path, sizeof(path), "%s/ckpt_%dx%d_t%g.bin", dir, g->Nx, g->Ny, tl[k]);
        FILE *fp = fopen(path, "wb"); if (!fp) die("ckpt open");
        const char magic[8] = "VDCKPT1";
        int hdr[4] = {g->Nx, g->Ny, step, VDC_NF};
        double dh[4] = {time, dtNext, g->Lx, g->Ly};
        fwrite(magic,1,8,fp); fwrite(hdr,sizeof(int),4,fp); fwrite(dh,sizeof(double),4,fp);
        const size_t ntot = (size_t)(g->Nx+2)*(size_t)(g->Ny+2);
        double *a[VDC_NF]; vdc_arrays(f, a);
        for (int q=0; q<VDC_NF; ++q) if (fwrite(a[q], sizeof(double), ntot, fp) != ntot) die("ckpt write");
        fclose(fp);
        fprintf(stderr, "CKPT wrote %s t=%.9g step=%d dtNext=%.6e\n", path, time, step, dtNext);
        ++k;
    }
}
/* bilinear interpolation of a coarse cell-centred field (interior cells only, clamped at the boundary) */
static double vdc_interp(const double *a, int ncx, int ncy, double xi, double yj) {
    /* xi, yj: coarse cell coordinates, cell i centred at xi = i (1..ncx) */
    int i0 = (int)floor(xi), j0 = (int)floor(yj);
    double wx = xi - i0, wy = yj - j0;
    if (i0 < 1) { i0 = 1; wx = 0.0; } if (i0 >= ncx) { i0 = ncx-1; wx = 1.0; }
    if (j0 < 1) { j0 = 1; wy = 0.0; } if (j0 >= ncy) { j0 = ncy-1; wy = 1.0; }
    if (ncx == 1) { i0 = 1; wx = 0.0; }
    if (ncy == 1) { j0 = 1; wy = 0.0; }
    const int i1 = (ncx == 1) ? i0 : i0+1, j1 = (ncy == 1) ? j0 : j0+1;
    return (1.0-wx)*(1.0-wy)*a[IDX(i0,j0,ncy)] + wx*(1.0-wy)*a[IDX(i1,j0,ncy)]
         + (1.0-wx)*wy*a[IDX(i0,j1,ncy)] + wx*wy*a[IDX(i1,j1,ncy)];
}
/* returns 1 for an exact (same-mesh) continuation, 0 for an interpolated start */
static int vdc_restart(const Grid *g, const Phys *ph, Fields *f, const char *path, double *time, double *dt, int *stepCkpt) {
    FILE *fp = fopen(path, "rb"); if (!fp) die("restart open");
    char magic[8]; int hdr[4]; double dh[4];
    if (fread(magic,1,8,fp)!=8 || strncmp(magic,"VDCKPT1",7)) die("restart: bad magic");
    if (fread(hdr,sizeof(int),4,fp)!=4 || fread(dh,sizeof(double),4,fp)!=4) die("restart: bad header");
    const int ncx = hdr[0], ncy = hdr[1];
    if (hdr[3] != VDC_NF) die("restart: field count");
    if (fabs(dh[2]-g->Lx) > 1e-12 || fabs(dh[3]-g->Ly) > 1e-12) die("restart: domain size differs");
    const size_t nc = (size_t)(ncx+2)*(size_t)(ncy+2);
    double *buf = malloc((size_t)VDC_NF*nc*sizeof(double)); if (!buf) die("restart alloc");
    if (fread(buf, sizeof(double), (size_t)VDC_NF*nc, fp) != (size_t)VDC_NF*nc) die("restart: short file");
    fclose(fp);
    *time = dh[0]; *stepCkpt = hdr[2];
    double *a[VDC_NF]; vdc_arrays(f, a);
    const int nx = g->Nx, ny = g->Ny;
    int exact = (ncx == nx && ncy == ny);
    if (exact) {
        for (int q=0; q<VDC_NF; ++q) memcpy(a[q], buf + (size_t)q*nc, nc*sizeof(double));
        *dt = dh[1];
    } else {
        if (nx < ncx || ny < ncy) die("restart: target mesh must not be coarser than the checkpoint");
        const double rx = (double)nx/ncx, ry = (double)ny/ncy;
        /* interpolate p,u,v,T,h,YF,YO,YP (fields 0..7); YN2, rho, fluxes are rebuilt below */
        for (int q=0; q<8; ++q) {
            const double *src = buf + (size_t)q*nc; double *dst = a[q];
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
                const double xi = (i-0.5)/rx + 0.5, yj = (j-0.5)/ry + 0.5;
                dst[IDX(i,j,ny)] = vdc_interp(src, ncx, ncy, xi, yj);
            }
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i=1; i<=nx; ++i) for (int j=1; j<=ny; ++j) {
            f->YF[IDX(i,j,ny)] = clamp(f->YF[IDX(i,j,ny)], 0.0, 1.0);
            f->YO[IDX(i,j,ny)] = clamp(f->YO[IDX(i,j,ny)], 0.0, 0.21);
            f->YP[IDX(i,j,ny)] = clamp(f->YP[IDX(i,j,ny)], 0.0, MAX(0.0, 1.0 - f->YF[IDX(i,j,ny)] - f->YO[IDX(i,j,ny)]));
            f->h [IDX(i,j,ny)] = MAX(0.0, f->h[IDX(i,j,ny)]);
        }
        /* diffusion-limited step: dt scales with the square of the cell size */
        *dt = dh[1] / (MAX(rx,ry)*MAX(rx,ry));
        apply_bc_velocity_pressure(g, ph, f);
        apply_bc_scalar_of(g, f, f->YF, ph->YF_fuel, ph->YF_ox, 0.0);
        apply_bc_scalar_of(g, f, f->YO, ph->YO_fuel, ph->YO_ox, 0.0);
        apply_bc_scalar_of(g, f, f->YP, ph->YP_fuel, ph->YP_ox, 0.0);
        apply_bc_h_exact(g, f->v, f->h);
        refresh_temperature_from_enthalpy(g, ph, f);
        update_thermo_transport_fields(g, ph, f, false);   /* rho from the equation of state */
        apply_bc_velocity_pressure(g, ph, f);
        compute_face_flux_rc(g, ph, f, *dt, 0);           /* same flux construction as a cold start */
    }
    if (exact) update_thermo_transport_fields(g, ph, f, false);  /* transport props (rho unchanged: EOS of restored state) */
    free(buf);
    fprintf(stderr, "RESTART from %s (%dx%d -> %dx%d, %s) t=%.9g ckptStep=%d dt=%.6e\n",
            path, ncx, ncy, nx, ny, exact ? "exact" : "interpolated", *time, *stepCkpt, *dt);
    return exact;
}
/* -------------------------------------------------------------------------------------------------- */

int main(int argc, char **argv) {
    struct timespec total_t0, total_t1;
    timer_now(&total_t0);

    Grid g; Phys ph; Controls c; Fields f;
    init_case(&g,&ph,&c);
    parse_args(argc,argv,&g,&ph,&c);

    omp_set_dynamic(0);
    omp_set_num_threads(c.threads);

    alloc_fields(&g,&f);
    init_fields(&g,&ph,&f);
    refresh_temperature_from_enthalpy(&g, &ph, &f);
    update_thermo_transport_fields(&g, &ph, &f, false);
    apply_bc_velocity_pressure(&g,&ph,&f);

    printf("Low-Mach reacting-flow solver with a multiple-coarse-grid RMT pressure solve\n");
    printf("Case ID: %s\n", c.caseId);
    printf("RMT pressure grid: finest %dx%d; automatic independent factor-3 hierarchy\n", g.Nx, g.Ny);
#ifdef RMT_ANISO
    printf("Pressure solver: RMTA (rmta_impl.h, x-only levels k=round(log3(dy/dx)))\n");
#else
    printf("Pressure solver: RMT (rmt_impl.h)\n");
#endif
    printf("RMT: smoothSweeps=%d, maxCycles=%d, threads=%d\n", c.mgPostSmooth, c.poissonIters, c.threads);
    printf("Grid Nx=%d Ny=%d dx=%.6e dy=%.6e\n",g.Nx,g.Ny,g.dx,g.dy);
    printf("Time end=%g dt0=%g dtMax=%g write=%g maxCo=%g maxDi=%g\n",
           c.endTime,c.dt0,c.dtMax,c.writeInterval,c.maxCo,c.maxDi);
    printf("Pressure maxSweeps=%d relTol=%.3e absTol=%.3e\n",
           c.poissonIters,c.pressureRelTol,c.pressureAbsTol);
    printf("Scalar cycles=%d sweepsPerCycle=%d relaxSc=%g relaxH=%g\n",
           c.scalarIters,c.scalarSweepsPerCycle,c.relaxSc,c.relaxH);
    printf("Chemistry=%s chemDt0=%g chemDtMax=%g dTchemMax=%g reactedFracMax=%g A=%g beta=%g Ta=%g\n",
           ph.chemistryOn ? "on" : "off",ph.initialChemicalTimeStep,ph.maxChemicalTimeStep,
           ph.dTchemMax,ph.reactedFracMax,ph.A,ph.beta,ph.Ta);
    printf("Thermo variableCp=%s sutherland=%s variableDensity=%s hydroPInRho=%s rhoRelax=%g referenceT=%g p0=%g\n",
           ph.variableCpOn ? "on" : "off", ph.sutherlandOn ? "on" : "off",
           ph.variableRhoOn ? "on" : "off", ph.hydroPressureInDensityOn ? "on" : "off",
           ph.rhoRelax, ph.Tref, ph.p0);
    printf("Reference properties rho=%g mu=%g cp=%g Pr=%g Sc=%g HfF=%g\n",
           ph.rho, ph.mu, ph.cp, ph.Pr, ph.Sc, ph.Hf_F);
    printf("Output=%s outputDir=%s; compute timer excludes VTK/PVD/CSV/performance I/O.\n",
           c.writeOutput ? "on" : "off", c.outputDir);

    /* Initial accepted face fluxes. */
    double vdcTime0 = c.startTime, vdcDt0 = MIN(c.dt0,c.dtMax); int vdcStep0 = 0;
    if (c.restartFile[0]) {
        vdc_restart(&g,&ph,&f,c.restartFile,&vdcTime0,&vdcDt0,&vdcStep0);
        vdcDt0 = MIN(vdcDt0, c.dtMax);
        c.startTime = vdcTime0;
        printf("Restart: file=%s startTime=%.9g dt=%.6e checkpointStep=%d\n", c.restartFile, vdcTime0, vdcDt0, vdcStep0);
    } else
    compute_face_flux_rc(&g,&ph,&f,MIN(c.dt0,c.dtMax),0);

    double time=c.startTime;
    double dt=vdcDt0;
    double nextWrite = (c.writeInterval > 0.0) ? c.startTime + c.writeInterval : HUGE_VAL;
    int step=0;
    double computeElapsed=0.0;
    double outputElapsed=0.0;
    double pressureElapsed=0.0;
    long long pressureSolves=0;
    long long totalPressureCycles=0;
    int maxPressureCycles=0;
    double sumConvergenceFactor=0.0;
    long long convergenceFactorCount=0;
    PressureStats pStats = {0,0.0,0.0,0.0,0.0,1.0};
    MassStats mStats = {0.0,0.0,0.0};
    CycleHistory cycleHistory = {0,0,NULL,NULL};
    PressureStepHistory stepHistory = {NULL,0,0};
    LowMachHistory lmHistory = {NULL,0,0};
    LowMachRecord lmRec; memset(&lmRec,0,sizeof(lmRec));
    LowMachChemBuf lmChem; memset(&lmChem,0,sizeof(lmChem));
    PvdWriter pvd = {NULL,{0}};
    double lastOutputTime = -HUGE_VAL;

    if (c.writeOutput) {
        struct timespec out_t0, out_t1;
        timer_now(&out_t0);
        pvd_open(&pvd, c.outputDir);
        write_vti(&g,&ph,&f,time,c.outputDir,&pvd);
        write_centerline_csv(&g,&f,time,c.outputDir);
            if (c.lowMachDiag) write_lowmach_centerline(&g,&f,time,c.outputDir);
        lastOutputTime = time;
        timer_now(&out_t1);
        outputElapsed += elapsed_seconds(&out_t0,&out_t1);
    }

    while (time < c.endTime - 1e-15) {
        struct timespec step_t0, step_t1;
        timer_now(&step_t0);

        ++step;
        if (time + dt > c.endTime) dt = c.endTime - time;
        const double dtUsed = dt;

        /* rhoOld is the accepted thermodynamic density at time n. */
        memcpy(f.rhoOld, f.rho, (size_t)(g.Nx+2)*(size_t)(g.Ny+2)*sizeof(double));

        apply_bc_velocity_pressure(&g,&ph,&f);
        apply_bc_scalar_of(&g,&f,f.YF,ph.YF_fuel,ph.YF_ox,0.0);
        apply_bc_scalar_of(&g,&f,f.YO,ph.YO_fuel,ph.YO_ox,0.0);
        apply_bc_scalar_of(&g,&f,f.YP,ph.YP_fuel,ph.YP_ox,0.0);
        apply_bc_h_exact(&g, f.v, f.h);
        refresh_temperature_from_enthalpy(&g, &ph, &f);

        /* Accepted conservative time-n mass fluxes drive transport and momentum. */
        momentum_predictor(&g,&ph,&f,dt,c.relaxU);

        memcpy(f.phiOld,f.YF,(size_t)(g.Nx+2)*(size_t)(g.Ny+2)*sizeof(double));
        build_scalar_rhs_openfoam_like(&g,&ph,&f,f.phiOld,f.rhsScalar,dt,ph.YF_fuel,ph.YF_ox,0.0);
        solve_scalar_helmholtz(&g,&ph,&f,f.YF,f.rhsScalar,dt,f.DMix,c.scalarIters,c.scalarSweepsPerCycle,
                               0.0,1.0,ph.YF_fuel,ph.YF_ox,0.0,c.relaxSc);

        memcpy(f.phiOld,f.YO,(size_t)(g.Nx+2)*(size_t)(g.Ny+2)*sizeof(double));
        build_scalar_rhs_openfoam_like(&g,&ph,&f,f.phiOld,f.rhsScalar,dt,ph.YO_fuel,ph.YO_ox,0.0);
        solve_scalar_helmholtz(&g,&ph,&f,f.YO,f.rhsScalar,dt,f.DMix,c.scalarIters,c.scalarSweepsPerCycle,
                               0.0,0.21,ph.YO_fuel,ph.YO_ox,0.0,c.relaxSc);

        memcpy(f.phiOld,f.YP,(size_t)(g.Nx+2)*(size_t)(g.Ny+2)*sizeof(double));
        build_scalar_rhs_openfoam_like(&g,&ph,&f,f.phiOld,f.rhsScalar,dt,ph.YP_fuel,ph.YP_ox,0.0);
        solve_scalar_helmholtz(&g,&ph,&f,f.YP,f.rhsScalar,dt,f.DMix,c.scalarIters,c.scalarSweepsPerCycle,
                               0.0,1.0,ph.YP_fuel,ph.YP_ox,0.0,c.relaxSc);

        memcpy(f.phiOld,f.h,(size_t)(g.Nx+2)*(size_t)(g.Ny+2)*sizeof(double));
        build_scalar_rhs_openfoam_like(&g,&ph,&f,f.phiOld,f.rhsScalar,dt,0.0,0.0,0.0);
        solve_scalar_helmholtz(&g,&ph,&f,f.h,f.rhsScalar,dt,f.alphaMix,c.scalarIters,c.scalarSweepsPerCycle,
                               0.0,1.0e12,0.0,0.0,0.0,c.relaxH);

        if (c.lowMachDiag) lowmach_chem_before(&g,&f,&lmChem);
        chemistry_subcycle(&g,&ph,&f,dt);
        if (c.lowMachDiag) lowmach_chem_after(&g,&ph,&f,&lmChem,dt,&lmRec);
        update_inert(&g,&f);
        refresh_temperature_from_enthalpy(&g, &ph, &f);
        update_thermo_transport_fields(&g, &ph, &f, true);

        compute_face_flux_rc(&g,&ph,&f,dt,1);
        g_pdumpTime = time + dt;
        build_pressure_rhs(&g,&ph,&f,dt);
        if (c.lowMachDiag) lowmach_pre(&g,&ph,&f,dt,&lmRec);
        {
            struct timespec p_t0, p_t1;
            timer_now(&p_t0);
            pStats = solve_pressure_poisson(&g,&ph,&f,&c,&cycleHistory);
            timer_now(&p_t1);
            pressureElapsed += elapsed_seconds(&p_t0,&p_t1);

            ++pressureSolves;
            totalPressureCycles += pStats.cycles;
            if (pStats.cycles > maxPressureCycles) maxPressureCycles = pStats.cycles;
            if (pStats.cycles > 0 && isfinite(pStats.convergenceFactor)) {
                sumConvergenceFactor += pStats.convergenceFactor;
                ++convergenceFactorCount;
            }
        }

        correct_velocity_and_face_flux(&g,&ph,&f,dt);
        apply_bc_velocity_pressure(&g,&ph,&f);
        mStats = compute_mass_stats(&g,&ph,&f,dt);
        if (c.lowMachDiag) {
            lowmach_post(&g,&ph,&f,dt,&lmRec);
            lowmach_global(&g,&ph,&f,dt,&lmChem,&lmRec);
            lmRec.step=step; lmRec.time=time+dt; lmRec.dt=dtUsed; lmRec.cycles=pStats.cycles;
            lowmach_append(&lmHistory,&lmRec);
        }

        double Co = compute_max_co(&g,&f,dt);
        double Di = compute_max_diffusion_number(&g,&f,dt);
        double dtCo = (Co > 1.0e-14) ? 0.95*c.maxCo*dt/Co : 1.2*dt;
        double dtDi = (Di > 1.0e-14) ? 0.95*c.maxDi*dt/Di : 1.2*dt;
        double dtGrow = 1.2*dt;
        double dtNext = MIN(c.dtMax, MIN(dtGrow, MIN(dtCo,dtDi)));
        if (!isfinite(dtNext) || dtNext <= 0.0) die("invalid adaptive timestep");

        time += dt;
        PressureStepRecord rec = {
            step, time, dtUsed, pStats.cycles, pStats.initialResidual,
            pStats.absResidual, pStats.relResidual, pStats.convergenceFactor
        };
        pressure_step_history_append(&stepHistory, &rec);

        timer_now(&step_t1);
        computeElapsed += elapsed_seconds(&step_t0,&step_t1);

        int writeNow = c.writeOutput && time >= nextWrite - 1e-14;
        int progressNow = (c.progressEvery > 0 && (step % c.progressEvery)==0)
                       || writeNow || time >= c.endTime - 1e-14;
        if (progressNow) {
            double Tmax=field_max(&g,f.T), YPmax=field_max(&g,f.YP);
            double YOmax=field_max(&g,f.YO), YFmax=field_max(&g,f.YF);
            printf("step=%7d t=%10.6f dt=%9.3e Co=%8.2e Di=%8.2e pSwp=%3d pRel=%9.2e rhoRBGS=%7.4f massMax=%9.2e massL2=%9.2e Tmax=%9.2f YPmax=%8.5f YOmax=%8.5f YFmax=%8.5f\n",
                   step,time,dtUsed,Co,Di,pStats.cycles,pStats.relResidual,pStats.convergenceFactor,
                   mStats.maxAbs,mStats.l2,Tmax,YPmax,YOmax,YFmax);
            fflush(stdout);
        }
        if (writeNow) {
            struct timespec out_t0, out_t1;
            timer_now(&out_t0);
            write_vti(&g,&ph,&f,time,c.outputDir,&pvd);
            write_centerline_csv(&g,&f,time,c.outputDir);
            if (c.lowMachDiag) write_lowmach_centerline(&g,&f,time,c.outputDir);
            lastOutputTime = time;
            timer_now(&out_t1);
            outputElapsed += elapsed_seconds(&out_t0,&out_t1);
            while (nextWrite <= time + 1e-14) nextWrite += c.writeInterval;
        }
        dt = dtNext;
        vdc_checkpoint_maybe(&g,&f,time,dt,step);
    }

    if (c.writeOutput) {
        struct timespec out_t0, out_t1;
        timer_now(&out_t0);
        if (fabs(time - lastOutputTime) > 1.0e-12*MAX(1.0,fabs(time))) {
            write_vti(&g,&ph,&f,time,c.outputDir,&pvd);
            write_centerline_csv(&g,&f,time,c.outputDir);
            if (c.lowMachDiag) write_lowmach_centerline(&g,&f,time,c.outputDir);
        }
        pvd_close(&pvd);
        write_pressure_step_history(&c, &stepHistory);
        if (c.lowMachDiag) write_lowmach_history(&c, &lmHistory);
        write_final_cycle_history(&c, &cycleHistory);
        timer_now(&out_t1);
        outputElapsed += elapsed_seconds(&out_t0,&out_t1);
    }

    timer_now(&total_t1);
    double totalWall = elapsed_seconds(&total_t0,&total_t1);
    double meanConvergenceFactor = convergenceFactorCount > 0
                                 ? sumConvergenceFactor/(double)convergenceFactorCount : 0.0;

    if (c.writeOutput) {
        struct timespec out_t0, out_t1;
        timer_now(&out_t0);
        write_case_summary_csv(&g,&ph,&c,&f,time,step,
                               computeElapsed,pressureElapsed,outputElapsed,totalWall,
                               pressureSolves,totalPressureCycles,maxPressureCycles,
                               meanConvergenceFactor,&pStats,&mStats);
        timer_now(&out_t1);
        outputElapsed += elapsed_seconds(&out_t0,&out_t1);
    }

    printf("Done. Final time=%g steps=%d\n",time,step);
    printf("Final pressure: cycles=%d initialResidual=%.6e absResidual=%.6e relResidual=%.6e rhsNorm=%.6e rhoRBGS=%.6f\n",
           pStats.cycles,pStats.initialResidual,pStats.absResidual,pStats.relResidual,pStats.rhsNorm,pStats.convergenceFactor);
    printf("Pressure totals: solves=%lld cycles=%lld avgCycles=%.6f maxCycles=%d pressureTime=%.6f s meanRhoRBGS=%.6f\n",
           pressureSolves,totalPressureCycles,
           pressureSolves ? (double)totalPressureCycles/(double)pressureSolves : 0.0,
           maxPressureCycles,pressureElapsed,meanConvergenceFactor);
    printf("RMT_DIAGNOSTICS levelsX=%d levelsY=%d direct_solves=%lld direct_unknowns=%lld lu_factorizations=%lld nonmonotone_cycles=%lld last_cycle_ratio=%.17g max_cycle_ratio=%.17g point_updates=%lld\n",
           g_rmtV2LastLevelX, g_rmtV2LastLevelY,
           g_rmtV2DirectSolvesTotal, g_rmtV2DirectUnknownsTotal, g_rmtV2LUFactorizationsTotal,
           g_rmtV2NonmonotoneCycles, g_rmtV2LastCycleRatio,
           g_rmtV2MaxCycleRatio, g_rmtV2PointUpdatesTotal);
    printf("Final mass residual: max=%.6e l2=%.6e integral=%.6e\n",
           mStats.maxAbs,mStats.l2,mStats.integral);
    printf("Compute time excluding output: %.6f s\n",computeElapsed);
    printf("Output write time: %.6f s\n",outputElapsed);
    printf("Total wall time including output/setup: %.6f s\n",totalWall);

    free(cycleHistory.absResidual);
    free(cycleHistory.relResidual);
    free(stepHistory.data);
    free(lmHistory.data);
    free(lmChem.YF); free(lmChem.YO); free(lmChem.YP); free(lmChem.h);
    free_fields(&f);
    return 0;
}
