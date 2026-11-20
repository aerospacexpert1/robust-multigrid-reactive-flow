/* mms.h -- manufactured solution of the MMS study (thesis Chapter 4).
   Operator: A p = -Laplacian(p), cell-centred five-point; boundary conditions of the production
   pressure-correction equation: p = 0 on x = 0, Lx (Dirichlet), dp/dy = 0 on y = 0, Ly (Neumann).
   p*(x,y) = sin(pi xi) cos(pi eta) + 0.5 sin(3 pi xi) cos(2 pi eta) + e^xi sin(pi xi) eta^2 (3 - 2 eta),
   xi = x/Lx, eta = y/Ly. Every term satisfies both boundary conditions exactly; the third term is not
   an eigenfunction of the discrete operator. */
#ifndef MMS_H
#define MMS_H
#include <math.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
static inline double mms_p(double x, double y, double Lx, double Ly) {
    const double xi=x/Lx, et=y/Ly, pi=M_PI;
    return sin(pi*xi)*cos(pi*et) + 0.5*sin(3*pi*xi)*cos(2*pi*et) + exp(xi)*sin(pi*xi)*et*et*(3.0-2.0*et);
}
static inline double mms_rhs(double x, double y, double Lx, double Ly) {   /* = -Laplacian(p*) */
    const double xi=x/Lx, et=y/Ly, pi=M_PI, ax=1.0/(Lx*Lx), ay=1.0/(Ly*Ly);
    double r = pi*pi*(ax+ay)*sin(pi*xi)*cos(pi*et)
             + 0.5*pi*pi*(9.0*ax+4.0*ay)*sin(3*pi*xi)*cos(2*pi*et);
    const double f=exp(xi)*sin(pi*xi), fpp=exp(xi)*((1.0-pi*pi)*sin(pi*xi)+2.0*pi*cos(pi*xi));
    const double g=et*et*(3.0-2.0*et), gpp=6.0-12.0*et;
    r += -(fpp*ax*g + f*gpp*ay);
    return r;
}
#endif
