
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define NPART_MAX 4000

int npart;                 
double x[NPART_MAX][3];    
double vh[NPART_MAX][3];   
double f[NPART_MAX][3];   

double side;                
double rcoff;                
double vir, epot;            
double ekin;                 
double vel;                    
double count;                   

double den   = 0.83134;    
double tref  = 0.722;      
double h     = 0.064;   
int    npartx = 8;       
int    nstep  = 20;         
int    nequil = 0;          

double a;                   

void init_lattice(int nside)
{
    int i, j, k, m, n;
    double xc, yc, zc;

    n = 0;
    for (i = 0; i < nside; i++) {
        for (j = 0; j < nside; j++) {
            for (k = 0; k < nside; k++) {
                xc = i * a;
                yc = j * a;
                zc = k * a;

                double basis[4][3] = {
                    {0.0,     0.0,     0.0},
                    {0.5*a,   0.5*a,   0.0},
                    {0.5*a,   0.0,     0.5*a},
                    {0.0,     0.5*a,   0.5*a}
                };

                for (m = 0; m < 4; m++) {
                    x[n][0] = xc + basis[m][0];
                    x[n][1] = yc + basis[m][1];
                    x[n][2] = zc + basis[m][2];
                    n++;
                }
            }
        }
    }
    npart = n;
}

void init_velocities(void)
{
    int i, k;
    double sp = sqrt(tref);
    double sumv[3] = {0.0, 0.0, 0.0};

    srand(12345);  

    for (i = 0; i < npart; i++) {
        for (k = 0; k < 3; k++) {
            double r = ((double)rand() / RAND_MAX) - 0.5;
            vh[i][k] = r * sp;
            sumv[k] += vh[i][k];
        }
    }

    for (k = 0; k < 3; k++) sumv[k] /= npart;
    for (i = 0; i < npart; i++)
        for (k = 0; k < 3; k++)
            vh[i][k] = (vh[i][k] - sumv[k]) * h;
}

void domove(void)
{
    int i, k;

    for (i = 0; i < npart; i++) {
        for (k = 0; k < 3; k++) {
            x[i][k] += vh[i][k] + f[i][k] * h * h;

            if (x[i][k] < 0.0)   x[i][k] += side;
            if (x[i][k] > side)  x[i][k] -= side;

            vh[i][k] += f[i][k] * h;   
        }
    }
}

void forces(void)
{
    int i, j;
    double xi, yi, zi, xx, yy, zz, rd;
    double rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148;
    double forcex, forcey, forcez;
    double sideh, rcoffs;

    vir = 0.0;
    epot = 0.0;
    sideh = side * 0.5;
    rcoffs = rcoff * rcoff;

    for (i = 0; i < npart; i++) {
        f[i][0] = 0.0;
        f[i][1] = 0.0;
        f[i][2] = 0.0;
    }

    #pragma omp parallel for \
        private(j, xi, yi, zi, xx, yy, zz, rd, rrd, rrd2, rrd3, rrd4, rrd6, rrd7, r148, forcex, forcey, forcez) \
        reduction(+:vir, epot) \
        schedule(static)
    for (i = 0; i < npart - 1; i++) {
        xi = x[i][0];
        yi = x[i][1];
        zi = x[i][2];

        for (j = i + 1; j < npart; j++) {
            xx = xi - x[j][0];
            yy = yi - x[j][1];
            zz = zi - x[j][2];

            if (xx < -sideh) xx += side;
            if (xx > sideh)  xx -= side;
            if (yy < -sideh) yy += side;
            if (yy > sideh)  yy -= side;
            if (zz < -sideh) zz += side;
            if (zz > sideh)  zz -= side;

            rd = xx*xx + yy*yy + zz*zz;

            if (rd <= rcoffs) {
                rrd  = 1.0 / rd;
                rrd2 = rrd * rrd;
                rrd3 = rrd2 * rrd;
                rrd4 = rrd2 * rrd2;
                rrd6 = rrd2 * rrd4;
                rrd7 = rrd6 * rrd;
                r148 = rrd7 - 0.5 * rrd4;

                forcex = xx * r148;
                forcey = yy * r148;
                forcez = zz * r148;

                f[i][0] += forcex;
                f[i][1] += forcey;
                f[i][2] += forcez;

                #pragma omp critical
                {
                    f[j][0] -= forcex;
                    f[j][1] -= forcey;
                    f[j][2] -= forcez;
                }

                vir  -= rd * r148;
                epot += rrd6 - rrd3;
            }
        }
    }
}

void mkekin(void)
{
    int i, k;
    double sum = 0.0;

    for (i = 0; i < npart; i++) {
        for (k = 0; k < 3; k++) {
            f[i][k] *= 48.0;
            vh[i][k] = (vh[i][k] + f[i][k] * h) * 0.5;
            sum += vh[i][k] * vh[i][k];
        }
    }
    ekin = sum / h;
}

void velavg(void)
{
    int i, k;
    double sv = 0.0, vscale, sp;

    sp = sqrt(tref) / sqrt(2.0 * ekin / (3.0 * npart));
    vscale = sp;

    for (i = 0; i < npart; i++) {
        for (k = 0; k < 3; k++) {
            vh[i][k] *= vscale;
            sv += fabs(vh[i][k]);
        }
    }
    vel = sv / (npart * h);
}

void prnout(int istep)
{
    double e, tscal;

    tscal = 2.0 * ekin / (3.0 * npart);
    e = (ekin + epot) / npart;

    printf("Step %4d  Temp = %10.6f  KE = %10.6f  PE = %10.6f  E/particle = %10.6f  Vel = %10.6f\n",
           istep, tscal, ekin, epot, e, vel);
}

int main(void)
{
    int istep;
    double tstart, tstop, tcalc;

    a = pow(4.0 / den, 1.0/3.0);
    side = a * npartx;
    rcoff = side * 0.5;   
    init_lattice(npartx);
    init_velocities();

    printf("Number of particles = %d\n", npart);
    printf("Box side length     = %f\n", side);
    printf("Threads used        = %d\n\n", omp_get_max_threads());

    tstart = omp_get_wtime();

    for (istep = 1; istep <= nstep; istep++) {
        domove();
        forces();
        mkekin();
        velavg();
        prnout(istep);
    }

    tstop = omp_get_wtime();
    tcalc = tstop - tstart;

    printf("\nTotal time: %f seconds\n", tcalc);

    return 0;
}