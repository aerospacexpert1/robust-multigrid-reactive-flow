/* mms_gen Nx Ny Lx Ly out.bin : writes the MMS-study system (format PSYS1) with the
   manufactured right-hand side b = -Laplacian(p*) of mms.h at the cell centres. */
#include <stdio.h>
#include <stdlib.h>
#include "mms.h"
int main(int argc,char**argv){
    if(argc<6){fprintf(stderr,"usage: mms_gen Nx Ny Lx Ly out.bin\n");return 2;}
    int Nx=atoi(argv[1]),Ny=atoi(argv[2]); double Lx=atof(argv[3]),Ly=atof(argv[4]);
    double dx=Lx/Nx,dy=Ly/Ny; size_t n=(size_t)Nx*Ny; double *b=malloc(n*sizeof(double)); double mx=0;
    for(int i=0;i<Nx;++i)for(int j=0;j<Ny;++j){double v=mms_rhs((i+0.5)*dx,(j+0.5)*dy,Lx,Ly); b[(size_t)i*Ny+j]=v; if(fabs(v)>mx)mx=fabs(v);}
    FILE*f=fopen(argv[5],"wb"); char magic[8]="PSYS1\0\0"; int ar=(int)lround(dy/dx); int hdr[3]={Nx,Ny,9000+ar}; double dh[3]={dx,dy,mx};
    fwrite(magic,1,8,f); fwrite(hdr,sizeof(int),3,f); fwrite(dh,sizeof(double),3,f); fwrite(b,sizeof(double),n,f); fwrite(b,sizeof(double),n,f); fclose(f);
    return 0;}
