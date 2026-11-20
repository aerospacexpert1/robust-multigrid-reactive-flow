/*
  pbench.c -- pressure benchmark code (PBENCH) of the thesis.

  Solves the pressure-correction equation  A e = r  (A = negative five-point Laplacian,
  cell-centred; Dirichlet on x = 0, Lx and Neumann on y = 0, Ly) for one stored system
  with a list of solvers, on the same system, with the same stopping test, in one run:

    RMT:s=k                      production RMT with k smoothing sweeps (include/rmt_impl.h)
    RMTA:s=k                     geometry-aware RMT (build with -DRMT_ANISO, include/rmta_impl.h)
    MG:r=2|3:g=1|2:L=0:c=direct  full-depth classical geometric multigrid, coarsening factor r,
                                 V-cycle (g=1) or W-cycle (g=2), exact banded Cholesky on the
                                 coarsest grid, 3+3 red-black Gauss-Seidel sweeps
    ...:sm=line                  same, zebra line Gauss-Seidel smoother along the strongly
                                 coupled direction (Thomas algorithm per line)
    ...:pre=n:post=m             number of pre- and post-smoothing sweeps

  The ten configurations of the thesis are
    RMT:s=16, RMTA:s=4, RMTA:s=6,
    MG:r=2:g=1:L=0:c=direct (MG2-V), MG:r=2:g=2:L=0:c=direct (MG2-W), MG:r=3:g=1:L=0:c=direct (MG3-V),
    MG:r=2:g=1:L=0:c=direct:sm=line              (MG2-V line (3,3)),
    MG:r=2:g=1:L=0:c=direct:sm=line:pre=1:post=1 (MG2-V line (1,1)),
    MG:r=2:g=2:L=0:c=direct:sm=line:pre=1:post=1 (MG2-W line (1,1)),
    MG:r=3:g=1:L=0:c=direct:sm=line              (MG3-V line (3,3)).

  Systems are written by solver/opposedflow.c (PDUMP_*; format PSYS2) or by mms_gen.c
  (manufactured solution of the MMS study; format PSYS1).
  Starts:  warm = recorded guess (previous-step pressure), p0 = uniform thermodynamic
           pressure (PSYS2 only), cold = zero.
  Tests:   prod   = production test  ||r||_inf <= max(1e-4 ||b||_inf, 1e-6)
           strict = ||r||_inf <= 1e-10 ||r_0||_inf
  If PBENCH_MMS_OUT is set, the L1/L2/Linf errors of the converged solution against the
  manufactured solution (mms.h) are appended to that file.

  Usage: pbench SYSTEM.bin warm|cold|p0 prod|strict MAXCYC OUT.csv HIST.csv SOLVER [SOLVER...]
*/
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#include "mms.h"

#define IDX(i,j,ny) ((i)*((ny)+2) + (j))
#define MAX(a,b) ((a)>(b)?(a):(b))
#define MIN(a,b) ((a)<(b)?(a):(b))

typedef struct { int Nx, Ny; double Lx, Ly, dx, dy; } Grid;
static void die(const char *m) { fprintf(stderr, "FATAL: %s\n", m); exit(1); }

#ifdef RMT_ANISO
#include "rmta_impl.h"
#define RMT_TAG "RMTA"
#else
#include "rmt_impl.h"
#define RMT_TAG "RMT"
#endif

static size_t gsize(const Grid *g) { return (size_t)(g->Nx+2)*(size_t)(g->Ny+2); }

/* ---------------- basic kernels (identical arithmetic to production) ---------------- */
static void bc_corr(const Grid *g, double *x) {
    const int nx=g->Nx, ny=g->Ny;
    for (int j=1;j<=ny;++j){ x[IDX(0,j,ny)]=-x[IDX(1,j,ny)]; x[IDX(nx+1,j,ny)]=-x[IDX(nx,j,ny)]; }
    for (int i=1;i<=nx;++i){ x[IDX(i,0,ny)]=x[IDX(i,1,ny)]; x[IDX(i,ny+1,ny)]=x[IDX(i,ny,ny)]; }
    x[IDX(0,0,ny)]=x[IDX(1,1,ny)]; x[IDX(0,ny+1,ny)]=x[IDX(1,ny,ny)];
    x[IDX(nx+1,0,ny)]=x[IDX(nx,1,ny)]; x[IDX(nx+1,ny+1,ny)]=x[IDX(nx,ny,ny)];
}
static void rbgs(const Grid *g, double *x, const double *b, int sweeps) {
    const int nx=g->Nx, ny=g->Ny;
    const double ax=1.0/(g->dx*g->dx), ay=1.0/(g->dy*g->dy), ap=2.0*ax+2.0*ay;
    for (int s=0;s<sweeps;++s) for (int color=0;color<2;++color) {
        bc_corr(g,x);
        #pragma omp parallel for schedule(static)
        for (int i=1;i<=nx;++i) {
            int j0 = 1 + ((i+1+color)&1);
            for (int j=j0;j<=ny;j+=2)
                x[IDX(i,j,ny)] = (b[IDX(i,j,ny)] + ax*(x[IDX(i+1,j,ny)]+x[IDX(i-1,j,ny)])
                                                 + ay*(x[IDX(i,j+1,ny)]+x[IDX(i,j-1,ny)]))/ap;
        }
    }
    bc_corr(g,x);
}
/* Zebra line Gauss-Seidel: lines along the strongly coupled direction (x if ax >= ay, else y),
   each line solved exactly (Thomas), lines of one colour in parallel. Boundary conditions are
   folded into the tridiagonal system: Dirichlet ends (ghost = -interior) and Neumann ends (ghost = +interior). */
static void line_gs(const Grid *g, double *x, const double *b, int sweeps) {
    const int nx=g->Nx, ny=g->Ny;
    const double ax=1.0/(g->dx*g->dx), ay=1.0/(g->dy*g->dy), ap=2.0*ax+2.0*ay;
    const int xlines = (ax >= ay);
    for (int s=0;s<sweeps;++s) for (int color=0;color<2;++color) {
        bc_corr(g,x);
        if (xlines) {
            /* Thomas factors depend only on i and on the line type (interior / j=1 / j=ny with Neumann);
               all lines of one colour are swept together, i outer and j inner (contiguous memory).
               dp is stored in place in x (the j+-1 neighbours belong to the other colour). */
            double *cp=malloc((size_t)3*(nx+2)*sizeof(double)), *mi=malloc((size_t)3*(nx+2)*sizeof(double));
            for (int t=0;t<3;++t) {
                double dg = ap - (t==1? ay:0.0) - (t==2? ay:0.0) - ((t==1 && ny==1)? ay:0.0);
                double *c=cp+(size_t)t*(nx+2), *m=mi+(size_t)t*(nx+2);
                for (int i=1;i<=nx;++i) {
                    double d=dg + ((i==1||i==nx)? ax:0.0);
                    double mm = d - (i>1? (-ax)*c[i-1]:0.0);
                    m[i]=1.0/mm; c[i]=(i<nx? -ax:0.0)*m[i];
                }
            }
            #pragma omp parallel
            {
              int nth=omp_get_num_threads(), tid=omp_get_thread_num();
              int nl=(ny-color+1)/2, a=(int)((long)nl*tid/nth), e=(int)((long)nl*(tid+1)/nth);
              for (int i=1;i<=nx;++i) {
                for (int k=a;k<e;++k) { int j=1+color+2*k;
                    int t = (j==1)? 1 : (j==ny? 2:0);
                    const double *c=cp+(size_t)t*(nx+2), *m=mi+(size_t)t*(nx+2); (void)c;
                    double rhs=b[IDX(i,j,ny)] + (j>1? ay*x[IDX(i,j-1,ny)]:0.0) + (j<ny? ay*x[IDX(i,j+1,ny)]:0.0);
                    x[IDX(i,j,ny)] = (rhs + (i>1? ax*x[IDX(i-1,j,ny)]:0.0))*m[i];
                }
              }
              for (int i=nx-1;i>=1;--i)
                for (int k=a;k<e;++k) { int j=1+color+2*k;
                    int t = (j==1)? 1 : (j==ny? 2:0);
                    x[IDX(i,j,ny)] -= cp[(size_t)t*(nx+2)+i]*x[IDX(i+1,j,ny)];
                }
            }
            free(cp); free(mi);
        } else {
            #pragma omp parallel
            { double *cp=malloc((size_t)(ny+2)*sizeof(double)), *dp=malloc((size_t)(ny+2)*sizeof(double));
              #pragma omp for schedule(static)
              for (int i=1+color;i<=nx;i+=2) {
                double dg=ap + ((i==1||i==nx)? ax:0.0);                   /* Dirichlet in x */
                for (int j=1;j<=ny;++j) {
                    double rhs=b[IDX(i,j,ny)] + ax*(x[IDX(i-1,j,ny)]+x[IDX(i+1,j,ny)]);
                    if (i==1) rhs-=ax*x[IDX(0,j,ny)]; if (i==nx) rhs-=ax*x[IDX(nx+1,j,ny)]; /* ghosts folded into dg */
                    double d=dg - ((j==1)? ay:0.0) - ((j==ny)? ay:0.0);    /* Neumann in y */
                    double lo=(j>1)? -ay:0.0, up=(j<ny)? -ay:0.0;
                    double m = d - (j>1? lo*cp[j-1]:0.0);
                    cp[j]=up/m; dp[j]=(rhs - (j>1? lo*dp[j-1]:0.0))/m;
                }
                x[IDX(i,ny,ny)]=dp[ny];
                for (int j=ny-1;j>=1;--j) x[IDX(i,j,ny)]=dp[j]-cp[j]*x[IDX(i,j+1,ny)];
              }
              free(cp); free(dp); }
        }
    }
    bc_corr(g,x);
}
static void resid(const Grid *g, const double *x, const double *b, double *r) {
    const int nx=g->Nx, ny=g->Ny;
    const double ax=1.0/(g->dx*g->dx), ay=1.0/(g->dy*g->dy), ap=2.0*ax+2.0*ay;
    #pragma omp parallel for schedule(static)
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j)
        r[IDX(i,j,ny)] = b[IDX(i,j,ny)] - (ap*x[IDX(i,j,ny)] - ax*(x[IDX(i+1,j,ny)]+x[IDX(i-1,j,ny)])
                                                         - ay*(x[IDX(i,j+1,ny)]+x[IDX(i,j-1,ny)]));
}
static double maxabs(const Grid *g, const double *a) {
    double m=0.0; const int nx=g->Nx, ny=g->Ny;
    #pragma omp parallel for reduction(max:m) schedule(static)
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j){ double v=fabs(a[IDX(i,j,ny)]); if(v>m) m=v; }
    return m;
}

/* ---------------- transfers ---------------- */
static void restrict_r(int r, const Grid *f, const double *rf, const Grid *c, double *bc) {
    memset(bc,0,gsize(c)*sizeof(double));
    const double w = 1.0/(double)(r*r);
    #pragma omp parallel for schedule(static)
    for (int ic=1;ic<=c->Nx;++ic) for (int jc=1;jc<=c->Ny;++jc) {
        const int i0=r*ic-(r-1), j0=r*jc-(r-1); double s=0.0;
        for (int a=0;a<r;++a) for (int q=0;q<r;++q) s += rf[IDX(i0+a,j0+q,f->Ny)];
        bc[IDX(ic,jc,c->Ny)] = s*w;
    }
}
static void prolong_add(int r, const Grid *c, double *ec, const Grid *f, double *xf) {
    bc_corr(c,ec);
    const int fy=f->Ny, cy=c->Ny;
    #pragma omp parallel for schedule(static)
    for (int ic=1;ic<=c->Nx;++ic) for (int jc=1;jc<=c->Ny;++jc) {
        const double cc=ec[IDX(ic,jc,cy)], cw=ec[IDX(ic-1,jc,cy)], ce=ec[IDX(ic+1,jc,cy)];
        const double cs=ec[IDX(ic,jc-1,cy)], cn=ec[IDX(ic,jc+1,cy)];
        const double csw=ec[IDX(ic-1,jc-1,cy)], cse=ec[IDX(ic+1,jc-1,cy)];
        const double cnw=ec[IDX(ic-1,jc+1,cy)], cne=ec[IDX(ic+1,jc+1,cy)];
        if (r==2) {
            const int il=2*ic-1, ir=2*ic, jb=2*jc-1, jt=2*jc;
            xf[IDX(il,jb,fy)] += 0.5625*cc+0.1875*cw+0.1875*cs+0.0625*csw;
            xf[IDX(ir,jb,fy)] += 0.5625*cc+0.1875*ce+0.1875*cs+0.0625*cse;
            xf[IDX(il,jt,fy)] += 0.5625*cc+0.1875*cw+0.1875*cn+0.0625*cnw;
            xf[IDX(ir,jt,fy)] += 0.5625*cc+0.1875*ce+0.1875*cn+0.0625*cne;
        } else {
            const int iL=3*ic-2, iC=3*ic-1, iR=3*ic, jB=3*jc-2, jC=3*jc-1, jT=3*jc;
            xf[IDX(iL,jC,fy)] += (2.0/3.0)*cc+(1.0/3.0)*cw;
            xf[IDX(iC,jC,fy)] += cc;
            xf[IDX(iR,jC,fy)] += (2.0/3.0)*cc+(1.0/3.0)*ce;
            xf[IDX(iL,jB,fy)] += (4.0/9.0)*cc+(2.0/9.0)*cw+(2.0/9.0)*cs+(1.0/9.0)*csw;
            xf[IDX(iC,jB,fy)] += (2.0/3.0)*cc+(1.0/3.0)*cs;
            xf[IDX(iR,jB,fy)] += (4.0/9.0)*cc+(2.0/9.0)*ce+(2.0/9.0)*cs+(1.0/9.0)*cse;
            xf[IDX(iL,jT,fy)] += (4.0/9.0)*cc+(2.0/9.0)*cw+(2.0/9.0)*cn+(1.0/9.0)*cnw;
            xf[IDX(iC,jT,fy)] += (2.0/3.0)*cc+(1.0/3.0)*cn;
            xf[IDX(iR,jT,fy)] += (4.0/9.0)*cc+(2.0/9.0)*ce+(2.0/9.0)*cn+(1.0/9.0)*cne;
        }
    }
}

/* ---------------- exact coarsest solve: banded Cholesky ---------------- */
typedef struct { int n, bw; double *L; } Band;
static void band_build(const Grid *g, Band *B) {
    const int nx=g->Nx, ny=g->Ny, n=nx*ny, bw=ny;
    const double ax=1.0/(g->dx*g->dx), ay=1.0/(g->dy*g->dy);
    B->n=n; B->bw=bw; B->L=calloc((size_t)n*(bw+1),sizeof(double));
    if(!B->L) die("band alloc");
    #define BL(i,k) B->L[(size_t)(i)*(bw+1)+(k)]   /* k = i-col, 0..bw */
    for (int i=1;i<=nx;++i) for (int j=1;j<=ny;++j) {
        const int row=(i-1)*ny+(j-1);
        double d=2.0*ax+2.0*ay;
        if (i==1) d+=ax; if (i==nx) d+=ax;      /* ghost = -interior (Dirichlet) */
        if (j==1) d-=ay; if (j==ny) d-=ay;      /* ghost = +interior (Neumann)   */
        BL(row,0)=d;
        if (j>1) BL(row,1)=-ay;
        if (i>1) BL(row,bw)=-ax;
    }
    for (int i=0;i<n;++i) {           /* in-place banded Cholesky, lower */
        for (int k=MAX(0,i-bw);k<i;++k) {
            double s=BL(i,i-k);
            for (int m=MAX(0,i-bw);m<k;++m) if (k-m<=bw) s-=BL(i,i-m)*BL(k,k-m);
            BL(i,i-k)=s/BL(k,0);
        }
        double s=BL(i,0);
        for (int m=MAX(0,i-bw);m<i;++m) s-=BL(i,i-m)*BL(i,i-m);
        if (!(s>0)) die("banded Cholesky failed");
        BL(i,0)=sqrt(s);
    }
    #undef BL
}
static void band_solve(const Grid *g, const Band *B, double *x, const double *b) {
    const int ny=g->Ny, n=B->n, bw=B->bw; double *y=malloc((size_t)n*sizeof(double));
    #define BL(i,k) B->L[(size_t)(i)*(bw+1)+(k)]
    for (int i=1;i<=g->Nx;++i) for (int j=1;j<=ny;++j) y[(i-1)*ny+(j-1)]=b[IDX(i,j,ny)];
    for (int i=0;i<n;++i){ double s=y[i]; for(int m=MAX(0,i-bw);m<i;++m) s-=BL(i,i-m)*y[m]; y[i]=s/BL(i,0); }
    for (int i=n-1;i>=0;--i){ double s=y[i]; for(int m=i+1;m<=MIN(n-1,i+bw);++m) s-=BL(m,m-i)*y[m]; y[i]=s/BL(i,0); }
    #undef BL
    for (int i=1;i<=g->Nx;++i) for (int j=1;j<=ny;++j) x[IDX(i,j,ny)]=y[(i-1)*ny+(j-1)];
    free(y); bc_corr(g,x);
}

/* ---------------- geometric multigrid hierarchy ---------------- */
#define MAXL 20
typedef struct {
    int r, gamma, L, coarseSweeps, direct, pre, post, line;
    Grid g[MAXL]; double *x[MAXL], *b[MAXL], *res[MAXL]; Band band;
} MG;
static void mg_setup(MG *m, const Grid *fine) {
    m->g[0]=*fine; int L=1;
    while (L < (m->L>0 ? m->L : MAXL)) {
        const Grid *p=&m->g[L-1];
        if (p->Nx % m->r || p->Ny % m->r || p->Nx/m->r < 2 || p->Ny/m->r < 2) break;
        Grid c=*p; c.Nx/=m->r; c.Ny/=m->r; c.dx*=m->r; c.dy*=m->r; m->g[L++]=c;
    }
    if (m->L>0 && L!=m->L) die("requested MG depth not available for this mesh");
    m->L=L;
    for (int l=0;l<L;++l){ size_t n=gsize(&m->g[l]);
        m->x[l]=calloc(n,sizeof(double)); m->b[l]=calloc(n,sizeof(double)); m->res[l]=calloc(n,sizeof(double)); }
    if (m->direct) band_build(&m->g[L-1], &m->band);
}
static void mg_cycle(MG *m, int l) {
    const Grid *g=&m->g[l];
    if (l==m->L-1) {
        if (m->direct) band_solve(g,&m->band,m->x[l],m->b[l]);
        else rbgs(g,m->x[l],m->b[l],m->coarseSweeps);
        return;
    }
    if (m->line) line_gs(g,m->x[l],m->b[l],m->pre); else rbgs(g,m->x[l],m->b[l],m->pre);
    resid(g,m->x[l],m->b[l],m->res[l]);
    restrict_r(m->r,g,m->res[l],&m->g[l+1],m->b[l+1]);
    memset(m->x[l+1],0,gsize(&m->g[l+1])*sizeof(double));
    const int visits = (l+1==m->L-1) ? 1 : m->gamma;   /* coarsest solved once per visit */
    for (int v=0; v<visits; ++v) mg_cycle(m,l+1);
    prolong_add(m->r,&m->g[l+1],m->x[l+1],g,m->x[l]);
    if (m->line) line_gs(g,m->x[l],m->b[l],m->post); else rbgs(g,m->x[l],m->b[l],m->post);
}

/* ---------------- solver driver ---------------- */
typedef struct { char name[64]; int kind; MG mg; int rmtSweeps; } Solver;   /* kind 0 SG,1 MG,2 RMT */
static void parse_solver(const char *spec, Solver *s) {
    memset(s,0,sizeof(*s)); snprintf(s->name,sizeof(s->name),"%s",spec);
    if (!strcmp(spec,"SG")) { s->kind=0; return; }
    if (!strncmp(spec,"RMT",3)) { if(strncmp(spec,RMT_TAG,strlen(RMT_TAG))||(!strcmp(RMT_TAG,"RMT")&&spec[3]=='A')) die("RMT variant not compiled in this binary"); s->kind=2; s->rmtSweeps=16; const char*q=strstr(spec,"s="); if(q) s->rmtSweeps=atoi(q+2); return; }
    if (!strncmp(spec,"MG",2)) {
        s->kind=1; MG *m=&s->mg; m->r=2; m->gamma=1; m->L=3; m->coarseSweeps=40; m->direct=0; m->pre=3; m->post=3;
        const char *q;
        if((q=strstr(spec,"r=")))  m->r=atoi(q+2);
        if((q=strstr(spec,"g=")))  m->gamma=atoi(q+2);
        if((q=strstr(spec,"L=")))  m->L=atoi(q+2);
        if((q=strstr(spec,"pre=")))  m->pre=atoi(q+4);
        if((q=strstr(spec,"post="))) m->post=atoi(q+5);
        if(strstr(spec,"sm=line")) m->line=1;
        if((q=strstr(spec,"c=direct"))) m->direct=1;
        else if((q=strstr(spec,"c=rbgs"))) m->coarseSweeps=atoi(q+6);
        return;
    }
    die("unknown solver spec");
}

typedef struct { int cycles, converged; double r0, rfin, time; } Result;

static Result run(Solver *s, const Grid *g, const double *r0, double rhsNorm, int strict, int maxcyc,
                  double *e, double *res, double *hist) {
    Result R={0,0,0,0,0};
    memset(e,0,gsize(g)*sizeof(double));
    memcpy(res,r0,gsize(g)*sizeof(double));
    double rn=maxabs(g,res); R.r0=rn; if(hist) hist[0]=rn;
    const double tolAbs = strict ? 1e-10*rn : 1e-6;
    const double tolRel = strict ? 0.0 : 1e-4*rhsNorm;
    double t0=omp_get_wtime();
    if (!(rn<=tolAbs || rn<=tolRel)) {
        RMTV2Workspace *ws = (s->kind==2) ? rmt_v2_workspace(g) : NULL;
        RMTV2Config cfg={0, s->rmtSweeps, 5};
        for (int k=0;k<maxcyc;++k) {
            if (s->kind==0) { rbgs(g,e,r0,1); }
            else if (s->kind==1) {
                MG *m=&s->mg;
                resid(g,e,r0,res);
                memcpy(m->b[0],res,gsize(g)*sizeof(double));
                memset(m->x[0],0,gsize(g)*sizeof(double));
                mg_cycle(m,0);
                #pragma omp parallel for schedule(static)
                for (int i=1;i<=g->Nx;++i) for (int j=1;j<=g->Ny;++j) e[IDX(i,j,g->Ny)]+=m->x[0][IDX(i,j,g->Ny)];
                bc_corr(g,e);
            } else {
                bc_corr(g,e); resid(g,e,r0,res);
                RMTV2Diag d; rmt_v2_build_correction(g,res,ws->corr,&cfg,&d);
                #pragma omp parallel for schedule(static)
                for (int i=1;i<=g->Nx;++i) for (int j=1;j<=g->Ny;++j) e[IDX(i,j,g->Ny)]+=ws->corr[IDX(i,j,g->Ny)];
                bc_corr(g,e);
            }
            resid(g,e,r0,res); rn=maxabs(g,res); R.cycles=k+1;
            if (hist && k+1<=maxcyc) hist[k+1]=rn;
            if (!isfinite(rn) || rn > 1e8*R.r0) break;   /* divergence guard */
            if (rn<=tolAbs || rn<=tolRel) { R.converged=1; break; }
        }
    } else R.converged=1;
    R.time=omp_get_wtime()-t0; R.rfin=rn;
    return R;
}

int main(int argc, char **argv) {
    if (argc<8) { fprintf(stderr,"usage: pbench SYS.bin warm|cold|p0 prod|strict MAXCYC OUT.csv HIST.csv SOLVER...\n"); return 2; }
    const char *sys=argv[1]; const int cold=!strcmp(argv[2],"cold"); const int p0s=!strcmp(argv[2],"p0");
    if (!cold && !p0s && strcmp(argv[2],"warm")) die("start must be warm, cold or p0"); const int strict=!strcmp(argv[3],"strict");
    const int maxcyc=atoi(argv[4]);
    FILE *fp=fopen(sys,"rb"); if(!fp) die("open system");
    char magic[8]; int hdr[3]; double dh[3];
    if (fread(magic,1,8,fp)!=8 || (strncmp(magic,"PSYS1",5) && strncmp(magic,"PSYS2",5))) die("bad magic");
    const int ndef = strncmp(magic,"PSYS2",5) ? 2 : 3;
    if (p0s && ndef < 3) die("p0 start needs a PSYS2 dump");
    if (fread(hdr,sizeof(int),3,fp)!=3 || fread(dh,sizeof(double),3,fp)!=3) die("bad header");
    Grid g={hdr[0],hdr[1],0,0,dh[0],dh[1]}; g.Lx=g.Nx*g.dx; g.Ly=g.Ny*g.dy;
    const int step=hdr[2]; const double rhsNorm=dh[2]; const size_t n=(size_t)g.Nx*g.Ny;
    double *buf=malloc((size_t)ndef*n*sizeof(double)); if(fread(buf,sizeof(double),(size_t)ndef*n,fp)!=(size_t)ndef*n) die("short system"); fclose(fp);
    double *r0=calloc(gsize(&g),sizeof(double)), *e=calloc(gsize(&g),sizeof(double)), *res=calloc(gsize(&g),sizeof(double));
    const double *src = buf + (p0s ? 2*n : (cold ? n : 0));
    for (int i=1;i<=g.Nx;++i) for (int j=1;j<=g.Ny;++j) r0[IDX(i,j,g.Ny)]=src[(size_t)(i-1)*g.Ny+(j-1)];
    double *hist=calloc((size_t)maxcyc+2,sizeof(double));
    FILE *out=fopen(argv[5],"a"), *hf=fopen(argv[6],"a"); if(!out||!hf) die("open outputs");
    const int threads=omp_get_max_threads();
    const char *SLAB = p0s ? "p0" : (cold ? "cold" : "warm");
    for (int a=7;a<argc;++a) {
        Solver s; parse_solver(argv[a],&s);
        int levels=0;
        if (s.kind==1) { mg_setup(&s.mg,&g); levels=s.mg.L; }
        if (s.kind==2) { RMTV2Config cfg={0,s.rmtSweeps,5}; rmt_v2_prepare_hierarchy(&g,&cfg); levels=rmt_v2_workspace(&g)->lm+1; }
        const int mc = (s.kind==0) ? (strict ? 2000000 : 500000) : maxcyc;
        double *h = (s.kind==0) ? NULL : hist;
        Result R=run(&s,&g,r0,rhsNorm,strict,mc,e,res,h);
        /* short solves: repeat for a stable timing (identical deterministic work) */
        int reps=1; double tsum=R.time;
        while (tsum<0.3 && reps<200 && R.cycles<=50) { Result Q=run(&s,&g,r0,rhsNorm,strict,mc,e,res,NULL); tsum+=Q.time; ++reps; }
        const double tavg=tsum/reps;
        const double rho = (R.cycles>0 && R.r0>0 && R.rfin>0) ? pow(R.rfin/R.r0,1.0/R.cycles) : 0.0;
        fprintf(out,"%s,%d,%d,%d,%s,%s,%s,%d,%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%d\n",
            sys,g.Nx,g.Ny,step,SLAB,strict?"strict":"prod",s.name,threads,levels,R.cycles,R.converged,
            R.r0,R.rfin,R.rfin/MAX(rhsNorm,1e-300),rho,tavg,R.cycles>0?tavg/R.cycles:0.0,reps);
        fflush(out);
        if (h && strict) { for (int k=0;k<=R.cycles;++k) fprintf(hf,"%d,%d,%d,%s,%s,%d,%d,%.6e\n",g.Nx,g.Ny,step,SLAB,s.name,threads,k,hist[k]); fflush(hf); }
        printf("%-34s %4dx%-4d step %-7d %-4s %-6s cyc=%-6d conv=%d rel=%.2e rho=%.3f t=%.4fs (%d reps)\n",
               s.name,g.Nx,g.Ny,step,SLAB,strict?"strict":"prod",R.cycles,R.converged,R.rfin/MAX(rhsNorm,1e-300),rho,tavg,reps);
        fflush(stdout);
        /* Benchmark A (MMS): discretisation error of the converged solution against p* */
        const char *mo=getenv("PBENCH_MMS_OUT");
        if (mo && *mo) {
            Result Q=run(&s,&g,r0,rhsNorm,strict,mc,e,res,NULL); (void)Q;
            double l2=0, li=0;
            for (int i=1;i<=g.Nx;++i) for (int j=1;j<=g.Ny;++j) {
                double d=e[IDX(i,j,g.Ny)]-mms_p((i-0.5)*g.dx,(j-0.5)*g.dy,g.Lx,g.Ly);
                l2+=d*d; if(fabs(d)>li) li=fabs(d);
            }
            l2=sqrt(l2/((double)g.Nx*g.Ny));
            FILE *mf=fopen(mo,"a"); if(!mf) die("open mms out");
            fprintf(mf,"%d,%d,%.6e,%.6e,%.4f,%s,%d,%d,%d,%.6e,%.6e,%.6e\n",g.Nx,g.Ny,g.dx,g.dy,g.dy/g.dx,s.name,threads,R.cycles,R.converged,l2,li,tavg);
            fclose(mf);
        }
    }
    fclose(out); fclose(hf); return 0;
}
