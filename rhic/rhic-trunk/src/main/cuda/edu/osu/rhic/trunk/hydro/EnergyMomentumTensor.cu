/*
 * EnergyMomentumTensor.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */
#include <math.h> // for math functions

#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"
#include "edu/osu/rhic/trunk/eos/EquationOfState.cuh"

#define MAX_ITERS 100000
//const PRECISION ACC = 1e-2;

__host__ __device__ 
PRECISION energyDensityFromConservedVariables(PRECISION ePrev, PRECISION M0, PRECISION M, PRECISION Pi) {
#ifndef CONFORMAL_EOS
	PRECISION e0 = ePrev;	// initial guess for energy density
	for(int j = 0; j < MAX_ITERS; ++j) {
		PRECISION p = equilibriumPressure(e0);
		PRECISION cs2 = speedOfSoundSquared(e0);
		PRECISION cst2 = p/e0;

		PRECISION A = fmaf(M0,1-cst2,Pi);
		PRECISION B = fmaf(M0,M0+Pi,-M);
		PRECISION H = sqrtf(fabsf(A*A+4*cst2*B));
		PRECISION D = (A-H)/(2*cst2);

		PRECISION f = e0 + D;
		PRECISION fp = 1 - ((cs2 - cst2)*(B + D*H - ((cs2 - cst2)*cst2*D*M0)/e0))/(cst2*e0*H);

		PRECISION e = e0 - f/fp;
		if(fabsf(e - e0) <=  0.001 * fabsf(e)) return e;
		e0 = e;
	}
//	printf("Maximum number of iterations exceeded.\n");
	printf("Maximum number of iterations exceeded.\tePrev=%.3f,\tM0=%.3f,\t M=%.3f,\t Pi=%.3f\n",ePrev,M0,M,Pi);
	return e0;
#else
	return fabsf(sqrtf(fabsf(4 * M0 * M0 - 3 * M)) - M0);
#endif
}

__host__ __device__ 
void getInferredVariables(PRECISION t, const PRECISION * const __restrict__ q, PRECISION ePrev,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p, 
PRECISION * const __restrict__ ut, PRECISION * const __restrict__ ux, PRECISION * const __restrict__ uy, PRECISION * const __restrict__ un
) {
	PRECISION ttt = q[0];
	PRECISION ttx = q[1];
	PRECISION tty = q[2];
	PRECISION ttn = q[3];
#ifdef PIMUNU
	PRECISION pitt = q[4];
	PRECISION pitx = q[5];
	PRECISION pity = q[6];
	PRECISION pitn = q[7];
#else
	PRECISION pitt = 0;
	PRECISION pitx = 0;
	PRECISION pity = 0;
	PRECISION pitn = 0;
#endif
	// \Pi
#ifdef PI
	PRECISION Pi = q[14];
#else
	PRECISION Pi = 0;
#endif

	PRECISION M0 = ttt - pitt;
	PRECISION M1 = ttx - pitx;
	PRECISION M2 = tty - pity;
	PRECISION M3 = ttn - pitn;
	PRECISION M = M1 * M1 + M2 * M2 + t * t * M3 * M3;
#ifdef Pi
	if ((M0 * M0 - M + M0 * Pi) < 0)
		Pi = M / M0 - M0;
#endif
/****************************************************************************/
//	if (ePrev <= 0.1) {
//		*e = M0 - M / M0;
//	} else {
		*e = energyDensityFromConservedVariables(ePrev, M0, M, Pi);
//	}
	if (isnan(*e)) {
		printf("M0=%.3f,\t M1=%.3f,\t M2=%.3f,\t M3=%.3f\n", M0, M1, M2, M3);
	}
	*p = equilibriumPressure(*e);
	if (*e < 1.e-7) {
		*e = 1.e-7;
		*p = 1.e-7;
	}

	PRECISION P = *p + Pi;
	PRECISION E = 1/(*e + P);
	*ut = sqrtf(fabsf((M0 + P) * E));
	PRECISION E2 = E/(*ut);
	*ux = M1 * E2;
	*uy = M2 * E2;
	*un = M3 * E2;
}

__global__ 
void setInferredVariablesKernel(const CONSERVED_VARIABLES * const __restrict__ q, 
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p, FLUID_VELOCITY * const __restrict__ u, 
PRECISION t
) {
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;

	if (threadID < d_nElements) {
		unsigned int k = threadID / (d_nx * d_ny) + N_GHOST_CELLS_M;
		unsigned int j = (threadID % (d_nx * d_ny)) / d_nx + N_GHOST_CELLS_M;
		unsigned int i = threadID % d_nx + N_GHOST_CELLS_M;
		unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		PRECISION q_s[NUMBER_CONSERVED_VARIABLES];
		q_s[0] = q->ttt[s];
		q_s[1] = q->ttx[s];
		q_s[2] = q->tty[s];
		q_s[3] = q->ttn[s];
#ifdef PIMUNU
		q_s[4] = q->pitt[s];
		q_s[5] = q->pitx[s];
		q_s[6] = q->pity[s];
		q_s[7] = q->pitn[s];
/****************************************************************************\
		q_s[8] = q->pixx[s];
		q_s[9] = q->pixy[s];
		q_s[10] = q->pixn[s];
		q_s[11] = q->piyy[s];
		q_s[12] = q->piyn[s];
		q_s[13] = q->pinn[s];
/****************************************************************************/
#endif
#ifdef PI
		q_s[14] = q->Pi[s];
#endif
		PRECISION _e, _p, ut, ux, uy, un;
		getInferredVariables(t, q_s, e[s], &_e, &_p, &ut, &ux, &uy, &un);
		e[s] = _e;
		p[s] = _p;
		u->ut[s] = ut;
		u->ux[s] = ux;
		u->uy[s] = uy;
		u->un[s] = un;
	}
}

//===================================================================
// Components of T^{\mu\nu} in (\tau,x,y,\eta_s)-coordinates
//===================================================================
__host__ __device__ 
PRECISION Ttt(PRECISION e, PRECISION p, PRECISION ut, PRECISION pitt) {
	return (e+p)*ut*ut-p+pitt;
}

__host__ __device__ 
PRECISION Ttx(PRECISION e, PRECISION p, PRECISION ut, PRECISION ux, PRECISION pitx) {
	return (e+p)*ut*ux+pitx;
}

__host__ __device__ 
PRECISION Tty(PRECISION e, PRECISION p, PRECISION ut, PRECISION uy, PRECISION pity) {
	return (e+p)*ut*uy+pity;
}

__host__ __device__ 
PRECISION Ttn(PRECISION e, PRECISION p, PRECISION ut, PRECISION un, PRECISION pitn) {
	return (e+p)*ut*un+pitn;
}

__host__ __device__ 
PRECISION Txx(PRECISION e, PRECISION p, PRECISION ux, PRECISION pixx) {
	return (e+p)*ux*ux+p+pixx;
}

__host__ __device__ 
PRECISION Txy(PRECISION e, PRECISION p, PRECISION ux, PRECISION uy, PRECISION pixy) {
	return (e+p)*ux*uy+pixy;
}

__host__ __device__ 
PRECISION Txn(PRECISION e, PRECISION p, PRECISION ux, PRECISION un, PRECISION pixn) {
	return (e+p)*ux*un+pixn;
}

__host__ __device__ 
PRECISION Tyy(PRECISION e, PRECISION p, PRECISION uy, PRECISION piyy) {
	return (e+p)*uy*uy+p+piyy;
}

__host__ __device__ 
PRECISION Tyn(PRECISION e, PRECISION p, PRECISION uy, PRECISION un, PRECISION piyn) {
	return (e+p)*uy*un+piyn;
}

__host__ __device__ 
PRECISION Tnn(PRECISION e, PRECISION p, PRECISION un, PRECISION pinn, PRECISION t) {
	return (e+p)*un*un+p/t/t+pinn;
}
