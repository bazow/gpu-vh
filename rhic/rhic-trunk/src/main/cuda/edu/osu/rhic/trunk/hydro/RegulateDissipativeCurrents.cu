#include <stdlib.h>
#include <stdio.h> // for printf

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"
#include "edu/osu/rhic/trunk/hydro/RegulateDissipativeCurrents.cuh"

#ifndef IDEAL
__global__ 
void regulateDissipativeCurrents(PRECISION t, 
CONSERVED_VARIABLES * const __restrict__ currrentVars, 
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u,
VALIDITY_DOMAIN * const __restrict__ validityDomain
) {
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadID < d_nElements) {
		unsigned int k = threadID / (d_nx * d_ny) + N_GHOST_CELLS_M;
		unsigned int j = (threadID % (d_nx * d_ny)) / d_nx + N_GHOST_CELLS_M;
		unsigned int i = threadID % d_nx + N_GHOST_CELLS_M;
		unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		PRECISION pitt = currrentVars->pitt[s];
		PRECISION pitx = currrentVars->pitx[s];
		PRECISION pity = currrentVars->pity[s];
		PRECISION pitn = currrentVars->pitn[s];
		PRECISION pixx = currrentVars->pixx[s];
		PRECISION pixy = currrentVars->pixy[s];
		PRECISION pixn = currrentVars->pixn[s];
		PRECISION piyy = currrentVars->piyy[s];
		PRECISION piyn = currrentVars->piyn[s];
		PRECISION pinn = currrentVars->pinn[s];
#ifdef Pi
		PRECISION Pi = currrentVars->Pi[s];
#else
		PRECISION Pi = 0;
#endif

		PRECISION ut = u->ut[s];
		PRECISION ux = u->ux[s];
		PRECISION uy = u->uy[s];
		PRECISION un = u->un[s];

		PRECISION e_s = e[s];
		PRECISION p_s = p[s];

		PRECISION xi0 = 0.1f;
		PRECISION rhomax = 1.0f;

xi0=1.0;
rhomax=10.0;

		PRECISION t2 = t*t;

		PRECISION pipi = pitt*pitt-2*pitx*pitx-2*pity*pity+pixx*pixx+2*pixy*pixy+piyy*piyy-2*pitn*pitn*t2+2*pixn*pixn*t2+2*piyn*piyn*t2+pinn*pinn*t2*t2;
		PRECISION spipi = sqrtf(fabsf(pipi+3*Pi*Pi));
		PRECISION pimumu = pitt - pixx - piyy - pinn*t*t;
		PRECISION piu0 = -pitn*t2*un + pitt*ut - pitx*ux - pity*uy;
		PRECISION piu1 = -pixn*t2*un + pitx*ut - pixx*ux - pixy*uy;
		PRECISION piu2 = -piyn*t2*un + pity*ut - pixy*ux - piyy*uy;
		PRECISION piu3 = -pinn*t2*un + pitn*ut - pixn*ux - piyn*uy;
		
		PRECISION a1 = fdividef(spipi, rhomax)*rsqrtf(e_s*e_s+3*p_s*p_s);
		PRECISION den = xi0*rhomax*spipi;
///*
		PRECISION a2 = fdividef(pimumu, den);
		PRECISION a3 = fdividef(piu0, den);
		PRECISION a4 = fdividef(piu1, den);
		PRECISION a5 = fdividef(piu2, den);
		PRECISION a6 = fdividef(piu3, den);
//*/
/*
		PRECISION a2 = fabsf(fdividef(pimumu, den));
		PRECISION a3 = fabsf(fdividef(piu0, den));
		PRECISION a4 = fabsf(fdividef(piu1, den));
		PRECISION a5 = fabsf(fdividef(piu2, den));
		PRECISION a6 = fabsf(fdividef(piu3, den));
//*/
		PRECISION rho = fmaxf(a1,fmaxf(a2,fmaxf(a3,fmaxf(a4,fmaxf(a5,a6)))));

		PRECISION fac = fdividef(tanhf(rho), rho);
		if(fabsf(rho)<1.e-7) fac = 1;

		currrentVars->pitt[s] *= fac;
		currrentVars->pitx[s] *= fac;
		currrentVars->pity[s] *= fac;
		currrentVars->pitn[s] *= fac;
		currrentVars->pixx[s] *= fac;
		currrentVars->pixy[s] *= fac;
		currrentVars->pixn[s] *= fac;
		currrentVars->piyy[s] *= fac;
		currrentVars->piyn[s] *= fac;
		currrentVars->pinn[s] *= fac;
		// TODO: Should we regulate \Pi here?

		validityDomain->regulations[s] = fac;
	}
}
#endif
