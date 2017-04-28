#include <stdlib.h>
#include <stdio.h> // for printf

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/trunk/hydro/FullyDiscreteKurganovTadmorScheme.cuh"
#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/core/muscl/SemiDiscreteKurganovTadmorScheme.cuh"
#include "edu/osu/rhic/core/muscl/HalfSiteExtrapolation.cuh"
#include "edu/osu//rhic/trunk/hydro/FluxFunctions.cuh"
#include "edu/osu//rhic/trunk/hydro/SpectralRadius.cuh"
#include "edu/osu/rhic/trunk/hydro/SourceTerms.cuh"
#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"
#include "edu/osu/rhic/trunk/hydro/RegulateDissipativeCurrents.cuh"

/**************************************************************************************************************************************************\
__device__ 
void setNeighborCells(const PRECISION * const __restrict__ data,
PRECISION * const __restrict__ I, PRECISION * const __restrict__ J, PRECISION * const __restrict__ K, PRECISION * const __restrict__ Q,
int s, unsigned int n, int ptr, int simm, int sim, int sip, int sipp, int sjmm, int sjm, int sjp, int sjpp, int skmm, int skm, int skp, int skpp) {
	PRECISION data_ns = data[s];
	// I
	*(I+ptr) = *(data+simm);
	*(I+ptr+1) = *(data+sim);
	*(I+ptr+2) = data_ns;
	*(I+ptr+3) = *(data+sip);
	*(I+ptr+4) = *(data+sipp);
	// J
	*(J+ptr) = *(data+sjmm);
	*(J+ptr+1) = *(data+sjm);
	*(J+ptr+2) = data_ns;
	*(J+ptr+3) = *(data+sjp);
	*(J+ptr+4) = *(data+sjpp);
	// K
	*(K+ptr) = *(data+skmm);
	*(K+ptr+1) = *(data+skm);
	*(K+ptr+2) = data_ns;
	*(K+ptr+3) = *(data+skp);
	*(K+ptr+4) = *(data+skpp);
	// Q
	*(Q + n) = data_ns;
}

__global__
void eulerStepKernel(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up
) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + N_GHOST_CELLS_M;
	int j = blockDim.y * blockIdx.y + threadIdx.y + N_GHOST_CELLS_M;
	int k = blockDim.z * blockIdx.z + threadIdx.z + N_GHOST_CELLS_M;

	PRECISION I[5 * NUMBER_CONSERVED_VARIABLES], J[5* NUMBER_CONSERVED_VARIABLES], K[5 * NUMBER_CONSERVED_VARIABLES];
	PRECISION hpx[NUMBER_CONSERVED_VARIABLES], hmx[NUMBER_CONSERVED_VARIABLES],
		hpy[NUMBER_CONSERVED_VARIABLES], hmy[NUMBER_CONSERVED_VARIABLES],
		hpz[NUMBER_CONSERVED_VARIABLES],	hmz[NUMBER_CONSERVED_VARIABLES];
	PRECISION Q[NUMBER_CONSERVED_VARIABLES], S[NUMBER_CONSERVED_VARIABLES];

	int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

	// calculate neighbor cell indices;
	int sim = s-1;
	int simm = sim-1;
	int sip = s+1;
	int sipp = sip+1;

	int sjm = s-d_ncx;
	int sjmm = sjm-d_ncx;
	int sjp = s+d_ncx;
	int sjpp = sjp+d_ncx;

	int stride = d_ncx * d_ncy;
	int skm = s-stride;
	int skmm = skm-stride;
	int skp = s+stride;
	int skpp = skp+stride;

	int ptr = 0;
	setNeighborCells(currrentVars->ttt,I,J,K,Q,s,0,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->ttx,I,J,K,Q,s,1,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->tty,I,J,K,Q,s,2,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->ttn,I,J,K,Q,s,3,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->pitt,I,J,K,Q,s,4,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->pitx,I,J,K,Q,s,5,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->pity,I,J,K,Q,s,6,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->pitn,I,J,K,Q,s,7,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->pixx,I,J,K,Q,s,8,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->pixy,I,J,K,Q,s,9,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->pixn,I,J,K,Q,s,10,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->piyy,I,J,K,Q,s,11,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->piyn,I,J,K,Q,s,12,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
	setNeighborCells(currrentVars->pinn,I,J,K,Q,s,13,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp);

	flux(I, hpx, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusX, &Fx, t, e[s]);
	flux(I, hmx, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusX, &Fx, t, e[s]);
	flux(J, hpy, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusY, &Fy, t, e[s]);
	flux(J, hmy, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusY, &Fy, t, e[s]);
	flux(K, hpz, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusZ, &Fz, t, e[s]);
	flux(K, hmz, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusZ, &Fz, t, e[s]);

	loadSourceTerms(I, J, K, Q, S, u, up->ut[s], up->ux[s], up->uy[s], up->un[s], t, e[s], p, s);

	PRECISION result[NUMBER_CONSERVED_VARIABLES];
	for (int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
		*(result+n) = *(Q+n) + d_dt * ( *(S+n) - ( *(hpx+n) - *(hmx+n) + *(hpy+n) - *(hmy+n) ) / d_dx - ( *(hpz+n) - *(hmz+n) )/d_dz );
	}

	updatedVars->ttt[s] = result[0];
	updatedVars->ttx[s] = result[1];
	updatedVars->tty[s] = result[2];
	updatedVars->ttn[s] = result[3];
	updatedVars->pitt[s] = result[4];
	updatedVars->pitx[s] = result[5];
	updatedVars->pity[s] = result[6];
	updatedVars->pitn[s] = result[7];
	updatedVars->pixx[s] = result[8];
	updatedVars->pixy[s] = result[9];
	updatedVars->pixn[s] = result[10];
	updatedVars->piyy[s] = result[11];
	updatedVars->piyn[s] = result[12];
	updatedVars->pinn[s] = result[13];
}
/**************************************************************************************************************************************************/

/**************************************************************************************************************************************************\
__device__ 
void setNeighborCells2(const PRECISION * const __restrict__ in, PRECISION * const __restrict__ out, PRECISION data_ns,
int ptr, int smm, int sm, int sp, int spp
) {
	*(out + ptr		) = in[smm];
	*(out + ptr + 1) = in[sm];
	*(out + ptr + 2) = data_ns;
	*(out + ptr + 3) = in[sp];
	*(out + ptr + 4) = in[spp];
}

__device__
void setNeighborCells3(const PRECISION * const __restrict__ in, PRECISION * const __restrict__ out, int ptr, int smm, int sm, int sp, int spp
) {
	*(out + ptr		) = in[smm];
	*(out + ptr + 1) = in[sm];
	*(out + ptr + 3) = in[sp];
	*(out + ptr + 4) = in[spp];
}

__global__
void eulerStepKernel_1D(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up
) {
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadID < d_nElements) {
		unsigned int k = threadID / (d_nx * d_ny) + N_GHOST_CELLS_M;
		unsigned int j = (threadID % (d_nx * d_ny)) / d_nx + N_GHOST_CELLS_M;
		unsigned int i = threadID % d_nx + N_GHOST_CELLS_M;
		unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		PRECISION I[5 * NUMBER_CONSERVED_VARIABLES], J[5* NUMBER_CONSERVED_VARIABLES], K[5 * NUMBER_CONSERVED_VARIABLES];
		PRECISION H[NUMBER_CONSERVED_VARIABLES], Q[NUMBER_CONSERVED_VARIABLES];

		// calculate neighbor cell indices;
		int sim = s-1;
		int simm = sim-1;
		int sip = s+1;
		int sipp = sip+1;

		int sjm = s-d_ncx;
		int sjmm = sjm-d_ncx;
		int sjp = s+d_ncx;
		int sjpp = sjp+d_ncx;

		int stride = d_ncx * d_ncy;
		int skm = s-stride;
		int skmm = skm-stride;
		int skp = s+stride;
		int skpp = skp+stride;

		int ptr = 0;
		setNeighborCells(currrentVars->ttt,I,J,K,Q,s,0,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->ttx,I,J,K,Q,s,1,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->tty,I,J,K,Q,s,2,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->ttn,I,J,K,Q,s,3,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->pitt,I,J,K,Q,s,4,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->pitx,I,J,K,Q,s,5,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->pity,I,J,K,Q,s,6,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->pitn,I,J,K,Q,s,7,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->pixx,I,J,K,Q,s,8,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->pixy,I,J,K,Q,s,9,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->pixn,I,J,K,Q,s,10,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->piyy,I,J,K,Q,s,11,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->piyn,I,J,K,Q,s,12,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->pinn,I,J,K,Q,s,13,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCells(currrentVars->Pi,I,J,K,Q,s,14,ptr,simm,sim,sip,sipp,sjmm,sjm,sjp,sjpp,skmm,skm,skp,skpp);

		//=======================================================================================================================
		PRECISION es = e[s];
		loadSourceTerms(I, J, K, Q, H, u, up->ut[s], up->ux[s], up->uy[s], up->un[s], t, es, p, s);

		PRECISION result[NUMBER_CONSERVED_VARIABLES];
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = *(Q+n) + d_dt * ( *(H+n) );
		}
		// X derivatives
		PRECISION facX = d_dt/d_dx;
		flux(I, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusX, &Fx, t, es);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) -= *(H+n)*facX;
		}
		flux(I, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusX, &Fx, t, es);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n)*facX;
		}
		// Y derivatives
		flux(I, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusY, &Fy, t, es);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) -= *(H+n)*facX;
		}
		flux(I, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusY, &Fy, t, es);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n)*facX;
		}
		// Z derivatives
		PRECISION facZ = d_dt/d_dz;
		flux(I, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusZ, &Fz, t, es);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) -= *(H+n)*facZ;
		}
		flux(I, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusZ, &Fz, t, es);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n)*facZ;
		}

		//=======================================================================================================================

		updatedVars->ttt[s] = result[0];
		updatedVars->ttx[s] = result[1];
		updatedVars->tty[s] = result[2];
		updatedVars->ttn[s] = result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] = result[4];
		updatedVars->pitx[s] = result[5];
		updatedVars->pity[s] = result[6];
		updatedVars->pitn[s] = result[7];
		updatedVars->pixx[s] = result[8];
		updatedVars->pixy[s] = result[9];
		updatedVars->pixn[s] = result[10];
		updatedVars->piyy[s] = result[11];
		updatedVars->piyn[s] = result[12];
		updatedVars->pinn[s] = result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] = result[14];
#endif
	}
}
/**************************************************************************************************************************************************/

/**************************************************************************************************************************************************/
__global__
void eulerStepKernelSource(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up
) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + N_GHOST_CELLS_M;
	int j = blockDim.y * blockIdx.y + threadIdx.y + N_GHOST_CELLS_M;
	int k = blockDim.z * blockIdx.z + threadIdx.z + N_GHOST_CELLS_M;

	if ( (i < d_ncx-2) && (j < d_ncy-2) && (k < d_ncz-2) ) {
		PRECISION Q[NUMBER_CONSERVED_VARIABLES];
		PRECISION S[NUMBER_CONSERVED_VARIABLES];

		int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		Q[0] = currrentVars->ttt[s];
		Q[1] = currrentVars->ttx[s];
		Q[2] = currrentVars->tty[s];
		Q[3] = currrentVars->ttn[s];
#ifdef PIMUNU
		Q[4] = currrentVars->pitt[s];
		Q[5] = currrentVars->pitx[s];
		Q[6] = currrentVars->pity[s];
		Q[7] = currrentVars->pitn[s];
		Q[8] = currrentVars->pixx[s];
		Q[9] = currrentVars->pixy[s];
		Q[10] = currrentVars->pixn[s];
		Q[11] = currrentVars->piyy[s];
		Q[12] = currrentVars->piyn[s];
		Q[13] = currrentVars->pinn[s];
#endif
#ifdef PI
		Q[14] = currrentVars->Pi[s];
#endif

		loadSourceTerms2(Q, S, u, up->ut[s], up->ux[s], up->uy[s], up->un[s], t, e[s], p, s);

		PRECISION result[NUMBER_CONSERVED_VARIABLES];
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = *(Q+n) + d_dt * ( *(S+n) );
		}

		updatedVars->ttt[s] = result[0];
		updatedVars->ttx[s] = result[1];
		updatedVars->tty[s] = result[2];
		updatedVars->ttn[s] = result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] = result[4];
		updatedVars->pitx[s] = result[5];
		updatedVars->pity[s] = result[6];
		updatedVars->pitn[s] = result[7];
		updatedVars->pixx[s] = result[8];
		updatedVars->pixy[s] = result[9];
		updatedVars->pixn[s] = result[10];
		updatedVars->piyy[s] = result[11];
		updatedVars->piyn[s] = result[12];
		updatedVars->pinn[s] = result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] = result[14];
#endif
	}
}
/**************************************************************************************************************************************************/

/**************************************************************************************************************************************************/
__device__
void setNeighborCellsJK2(const PRECISION * const __restrict__ in, PRECISION * const __restrict__ out,
int s, int ptr, int smm, int sm, int sp, int spp
) {
	PRECISION data_ns = in[s];
	*(out + ptr		) = in[smm];
	*(out + ptr + 1) = in[sm];
	*(out + ptr + 2) = data_ns;
	*(out + ptr + 3) = in[sp];
	*(out + ptr + 4) = in[spp];
}

__global__
void eulerStepKernelX(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const PRECISION * const __restrict__ e
) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + N_GHOST_CELLS_M;
	int j = blockDim.y * blockIdx.y + threadIdx.y + N_GHOST_CELLS_M;
	int k = blockDim.z * blockIdx.z + threadIdx.z + N_GHOST_CELLS_M;

	if ( (i < d_ncx-2) && (j < d_ncy-2) && (k < d_ncz-2) ) {
		PRECISION I[5 * NUMBER_CONSERVED_VARIABLES];
		PRECISION H[NUMBER_CONSERVED_VARIABLES];

		int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		// calculate neighbor cell indices;
		int sim = s-1;
		int simm = sim-1;
		int sip = s+1;
		int sipp = sip+1;

		int ptr=0;
		setNeighborCellsJK2(currrentVars->ttt,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttx,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->tty,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
#ifdef PIMUNU
		setNeighborCellsJK2(currrentVars->pitt,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitx,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pity,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixx,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixy,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyy,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pinn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
#endif
#ifdef PI
		setNeighborCellsJK2(currrentVars->Pi,I,s,ptr,simm,sim,sip,sipp);
#endif

		PRECISION result[NUMBER_CONSERVED_VARIABLES];
		flux(I, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusX, &Fx, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = - *(H+n);
		}
		flux(I, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusX, &Fx, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n);
			*(result+n) /= d_dx;
		}
#ifndef IDEAL
		loadSourceTermsX(I, H, u, s);
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) += *(H+n);
			*(result+n) *= d_dt;
		}
#else
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) *= d_dt;
		}
#endif	
		for (unsigned int n = 4; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) *= d_dt;
		}

		updatedVars->ttt[s] += result[0];
		updatedVars->ttx[s] += result[1];
		updatedVars->tty[s] += result[2];
		updatedVars->ttn[s] += result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] += result[4];
		updatedVars->pitx[s] += result[5];
		updatedVars->pity[s] += result[6];
		updatedVars->pitn[s] += result[7];
		updatedVars->pixx[s] += result[8];
		updatedVars->pixy[s] += result[9];
		updatedVars->pixn[s] += result[10];
		updatedVars->piyy[s] += result[11];
		updatedVars->piyn[s] += result[12];
		updatedVars->pinn[s] += result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] += result[14];
#endif
	}
}

__global__
void eulerStepKernelY(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const PRECISION * const __restrict__ e
) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + N_GHOST_CELLS_M;
	int j = blockDim.y * blockIdx.y + threadIdx.y + N_GHOST_CELLS_M;
	int k = blockDim.z * blockIdx.z + threadIdx.z + N_GHOST_CELLS_M;

	if ( (i < d_ncx-2) && (j < d_ncy-2) && (k < d_ncz-2) ) {
		PRECISION J[5* NUMBER_CONSERVED_VARIABLES];
		PRECISION H[NUMBER_CONSERVED_VARIABLES];

		int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		// calculate neighbor cell indices;
		int sjm = s-d_ncx;
		int sjmm = sjm-d_ncx;
		int sjp = s+d_ncx;
		int sjpp = sjp+d_ncx;

		int ptr=0;
		setNeighborCellsJK2(currrentVars->ttt,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttx,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->tty,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
#ifdef PIMUNU
		setNeighborCellsJK2(currrentVars->pitt,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitx,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pity,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixx,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixy,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyy,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pinn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
#endif
#ifdef PI
		setNeighborCellsJK2(currrentVars->Pi,J,s,ptr,sjmm,sjm,sjp,sjpp);
#endif

		PRECISION result[NUMBER_CONSERVED_VARIABLES];
		flux(J, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusY, &Fy, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = - *(H+n);
		}
		flux(J, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusY, &Fy, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n);
			*(result+n) /= d_dy;
		}
#ifndef IDEAL
		loadSourceTermsY(J, H, u, s);
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) += *(H+n);
			*(result+n) *= d_dt;
		}
#else
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) *= d_dt;
		}
#endif
		for (unsigned int n = 4; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) *= d_dt;
		}

		updatedVars->ttt[s] += result[0];
		updatedVars->ttx[s] += result[1];
		updatedVars->tty[s] += result[2];
		updatedVars->ttn[s] += result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] += result[4];
		updatedVars->pitx[s] += result[5];
		updatedVars->pity[s] += result[6];
		updatedVars->pitn[s] += result[7];
		updatedVars->pixx[s] += result[8];
		updatedVars->pixy[s] += result[9];
		updatedVars->pixn[s] += result[10];
		updatedVars->piyy[s] += result[11];
		updatedVars->piyn[s] += result[12];
		updatedVars->pinn[s] += result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] += result[14];
#endif
	}
}

__global__
void eulerStepKernelZ(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const PRECISION * const __restrict__ e
) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + N_GHOST_CELLS_M;
	int j = blockDim.y * blockIdx.y + threadIdx.y + N_GHOST_CELLS_M;
	int k = blockDim.z * blockIdx.z + threadIdx.z + N_GHOST_CELLS_M;

	if ( (i < d_ncx-2) && (j < d_ncy-2) && (k < d_ncz-2) ) {
//printf("(i,j,k)=(%d,%d,%d)\n",i,j,k);
		PRECISION K[5 * NUMBER_CONSERVED_VARIABLES];
		PRECISION H[NUMBER_CONSERVED_VARIABLES];

		int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		// calculate neighbor cell indices;
		int stride = d_ncx * d_ncy;
		int skm = s-stride;
		int skmm = skm-stride;
		int skp = s+stride;
		int skpp = skp+stride;

		int ptr=0;
		setNeighborCellsJK2(currrentVars->ttt,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttx,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->tty,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
#ifdef PIMUNU
		setNeighborCellsJK2(currrentVars->pitt,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitx,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pity,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixx,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixy,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyy,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pinn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
#endif
#ifdef PI
		setNeighborCellsJK2(currrentVars->Pi,K,s,ptr,skmm,skm,skp,skpp);
#endif

		PRECISION result[NUMBER_CONSERVED_VARIABLES];
		flux(K, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusZ, &Fz, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = -*(H+n);
		}
		flux(K, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusZ, &Fz, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n);
			*(result+n) /= d_dz;
		}
#ifndef IDEAL
		loadSourceTermsZ(K, H, u, s, t);
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) += *(H+n);
			*(result+n) *= d_dt;
		}
#else
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) *= d_dt;
		}
#endif
		for (unsigned int n = 4; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) *= d_dt;
		}

		updatedVars->ttt[s] += result[0];
		updatedVars->ttx[s] += result[1];
		updatedVars->tty[s] += result[2];
		updatedVars->ttn[s] += result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] += result[4];
		updatedVars->pitx[s] += result[5];
		updatedVars->pity[s] += result[6];
		updatedVars->pitn[s] += result[7];
		updatedVars->pixx[s] += result[8];
		updatedVars->pixy[s] += result[9];
		updatedVars->pixn[s] += result[10];
		updatedVars->piyy[s] += result[11];
		updatedVars->piyn[s] += result[12];
		updatedVars->pinn[s] += result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] += result[14];
#endif
	}
}

/**************************************************************************************************************************************************\

/**************************************************************************************************************************************************\
__device__
void setSharedData(const CONSERVED_VARIABLES * const __restrict__ in, PRECISION * const __restrict__ out,
int i, int j, int k, int nx, int ny, int stride, int s) {
	int ptr = i + nx*(j + ny*k);
	*(out + ptr) = in->ttt[s]; 	ptr+=stride;
	*(out + ptr) = in->ttx[s]; 	ptr+=stride;
	*(out + ptr) = in->tty[s]; 	ptr+=stride;
	*(out + ptr) = in->ttn[s]; 	ptr+=stride;
	*(out + ptr) = in->pitt[s]; 	ptr+=stride;
	*(out + ptr) = in->pitx[s]; 	ptr+=stride;
	*(out + ptr) = in->pity[s]; 	ptr+=stride;
	*(out + ptr) = in->pitn[s]; 	ptr+=stride;
	*(out + ptr) = in->pixx[s]; 	ptr+=stride;
	*(out + ptr) = in->pixy[s]; 	ptr+=stride;
	*(out + ptr) = in->pixn[s]; 	ptr+=stride;
	*(out + ptr) = in->piyy[s]; 	ptr+=stride;
	*(out + ptr) = in->piyn[s]; 	ptr+=stride;
	*(out + ptr) = in->pinn[s]; 	ptr+=stride;
#ifdef PI
	*(out + ptr) = in->Pi[s];
#endif
}


__global__
void eulerStepKernelSharedX(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const PRECISION * const __restrict__ e
) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + N_GHOST_CELLS_M;
	int j = blockDim.y * blockIdx.y + threadIdx.y + N_GHOST_CELLS_M;
	int k = blockDim.z * blockIdx.z + threadIdx.z + N_GHOST_CELLS_M;

	if ( (i < d_ncx-2) && (j < d_ncy-2) && (k < d_ncz-2) ) {
		__shared__ PRECISION s_data[NUMBER_CONSERVED_VARIABLES*(BSX_X+4)*BSX_Y*BSX_Z];

		int tx = threadIdx.x + N_GHOST_CELLS_M;
		int ty = threadIdx.y;
		int tz = threadIdx.z;
		int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		int stride = (BSX_X+4)*BSX_Y*BSX_Z;
		int ts = tx + (BSX_X+4) * (ty + BSX_Y * tz);

		setSharedData(currrentVars, s_data, tx, ty, tz, BSX_X+4, BSX_Y, stride, s);

		if(threadIdx.x==0) {
			int sim = s-1;
			int simm = sim-1;
			setSharedData(currrentVars, s_data, tx-1, ty, tz, BSX_X+4, BSX_Y, stride, sim);
			setSharedData(currrentVars, s_data, tx-2, ty, tz, BSX_X+4, BSX_Y, stride, simm);
		}
		if(threadIdx.x==blockDim.x-1) {
			int sip = s+1;
			int sipp = sip+1;
			setSharedData(currrentVars, s_data, tx+1, ty, tz, BSX_X+4, BSX_Y, stride, sip);
			setSharedData(currrentVars, s_data, tx+2, ty, tz, BSX_X+4, BSX_Y, stride, sipp);
		}

		__syncthreads();
		//==========================================================================================

		PRECISION e_s = e[s];
		PRECISION result[NUMBER_CONSERVED_VARIABLES], H[NUMBER_CONSERVED_VARIABLES];
		flux2(s_data, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusX, &Fx, t, ts, stride, e_s);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = - *(H+n);
		}
		flux2(s_data, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusX, &Fx, t, ts, stride, e_s);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n);
			*(result+n) /= d_dx;
		}
		// source
		PRECISION facX = 1/d_dx/2;
		int ptr = ts+4*stride;
		PRECISION dxpitt = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facX;	ptr+=stride;
		PRECISION dxpitx = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facX;	ptr+=stride;
		PRECISION dxpity = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facX;	ptr+=stride;
		PRECISION dxpitn = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facX;	ptr+=stride;
		PRECISION dxpixx = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facX;	ptr+=stride;
		PRECISION dxpixy = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facX;	ptr+=stride;
		PRECISION dxpixn = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facX;	ptr+=4*stride;

		PRECISION ut = u->ut[s];
		PRECISION ux = u->ux[s];
		PRECISION vx = fdividef(ux, ut);
#ifndef PI
		result[0] += dxpitt*vx - dxpitx;
		result[1] += dxpitx*vx - dxpixx;
#else
		PRECISION dxPi = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facX;
		result[0] += dxpitt*vx - dxpitx - vx*dxPi;
		result[1] += dxpitx*vx - dxpixx - dxPi;
#endif
		result[2] += dxpity*vx - dxpixy;
		result[3] += dxpitn*vx - dxpixn;

		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) *= d_dt;
		}
		//==========================================================================================

		updatedVars->ttt[s] += result[0];
		updatedVars->ttx[s] += result[1];
		updatedVars->tty[s] += result[2];
		updatedVars->ttn[s] += result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] += result[4];
		updatedVars->pitx[s] += result[5];
		updatedVars->pity[s] += result[6];
		updatedVars->pitn[s] += result[7];
		updatedVars->pixx[s] += result[8];
		updatedVars->pixy[s] += result[9];
		updatedVars->pixn[s] += result[10];
		updatedVars->piyy[s] += result[11];
		updatedVars->piyn[s] += result[12];
		updatedVars->pinn[s] += result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] += result[14];
#endif
	}
}

__global__
//__launch_bounds__(1024, 3)
void eulerStepKernelSharedY(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const PRECISION * const __restrict__ e
) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + N_GHOST_CELLS_M;
	int j = blockDim.y * blockIdx.y + threadIdx.y + N_GHOST_CELLS_M;
	int k = blockDim.z * blockIdx.z + threadIdx.z + N_GHOST_CELLS_M;

	if ( (i < d_ncx-2) && (j < d_ncy-2) && (k < d_ncz-2) ) {
		__shared__ PRECISION s_data[NUMBER_CONSERVED_VARIABLES*(BSY_Y+4)*BSY_X*BSY_Z];

		int tx = threadIdx.x;
		int ty = threadIdx.y + N_GHOST_CELLS_M;
		int tz = threadIdx.z;
		int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		int stride = (BSY_Y+4)*BSY_X*BSY_Z;
		int ts = ty + (BSY_Y+4) * (tx + BSY_X * tz);

		setSharedData(currrentVars, s_data, ty, tx, tz, BSY_Y+4, BSY_X, stride, s);

		if(threadIdx.y==0) {
			int sjm = s-d_ncx;
			int sjmm = sjm-d_ncx;
			setSharedData(currrentVars, s_data, ty-1, tx, tz, BSY_Y+4, BSY_X, stride, sjm);
			setSharedData(currrentVars, s_data, ty-2, tx, tz, BSY_Y+4, BSY_X, stride, sjmm);
		}
		if(threadIdx.y==blockDim.y-1) {
			int sjp = s+d_ncx;
			int sjpp = sjp+d_ncx;
			setSharedData(currrentVars, s_data, ty+1, tx, tz, BSY_Y+4, BSY_X, stride, sjp);
			setSharedData(currrentVars, s_data, ty+2, tx, tz, BSY_Y+4, BSY_X, stride, sjpp);
		}

		__syncthreads();
		//==========================================================================================

		PRECISION e_s = e[s];
		PRECISION result[NUMBER_CONSERVED_VARIABLES], H[NUMBER_CONSERVED_VARIABLES];
		flux2(s_data, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusY, &Fy, t, ts, stride, e_s);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = - *(H+n);
		}
		flux2(s_data, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusY, &Fy, t, ts, stride, e_s);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n);
			*(result+n) /= d_dy;
		}
		// source
		PRECISION facY = 1/d_dy/2;
		int ptr = ts+4*stride;
		PRECISION dypitt = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facY;	ptr+=stride;
		PRECISION dypitx = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facY;	ptr+=stride;
		PRECISION dypity = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facY;	ptr+=stride;
		PRECISION dypitn = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facY;	ptr+=stride; ptr+=stride;
		PRECISION dypixy = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facY;	ptr+=stride; ptr+=stride;
		PRECISION dypiyy = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facY;	ptr+=stride;
		PRECISION dypiyn = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facY;	ptr+=stride; ptr+=stride;

		PRECISION ut = u->ut[s];
		PRECISION uy = u->uy[s];
		PRECISION vy =fdividef(uy, ut);
#ifndef PI
		result[0] += dypitt*vy - dypity;
		result[2] += dypity*vy - dypiyy;
#else
		PRECISION dyPi = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facY;
		result[0] += dypitt*vy - dypity - vy*dyPi;
		result[2] += dypity*vy - dypiyy - dyPi;
#endif
		result[1] += dypitx*vy - dypixy;
		result[3] += dypitn*vy - dypiyn;

		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) *= d_dt;
		}
		//==========================================================================================

		updatedVars->ttt[s] += result[0];
		updatedVars->ttx[s] += result[1];
		updatedVars->tty[s] += result[2];
		updatedVars->ttn[s] += result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] += result[4];
		updatedVars->pitx[s] += result[5];
		updatedVars->pity[s] += result[6];
		updatedVars->pitn[s] += result[7];
		updatedVars->pixx[s] += result[8];
		updatedVars->pixy[s] += result[9];
		updatedVars->pixn[s] += result[10];
		updatedVars->piyy[s] += result[11];
		updatedVars->piyn[s] += result[12];
		updatedVars->pinn[s] += result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] += result[14];
#endif
	}
}

__global__
void eulerStepKernelSharedZ(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const PRECISION * const __restrict__ e
) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + N_GHOST_CELLS_M;
	int j = blockDim.y * blockIdx.y + threadIdx.y + N_GHOST_CELLS_M;
	int k = blockDim.z * blockIdx.z + threadIdx.z + N_GHOST_CELLS_M;

	if ( (i < d_ncx-2) && (j < d_ncy-2) && (k < d_ncz-2) ) {
		printf("(i,j,k)=(%d,%d,%d)\n",i,j,k);
		__shared__ PRECISION s_data[NUMBER_CONSERVED_VARIABLES*(BSZ_Z+4)*BSZ_X*BSZ_Y];

		int tx = threadIdx.x;
		int ty = threadIdx.y;
		int tz = threadIdx.z + N_GHOST_CELLS_M;
		int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		int stride = (BSZ_Z+4)*BSZ_X*BSZ_Y;
		int ts = tz + (BSZ_Z+4) * (ty + BSZ_Y * tx);

		setSharedData(currrentVars, s_data, tz, ty, tx, BSZ_Z+4, BSZ_Y, stride, s);

		if(threadIdx.z==0) {
			int skm = s-d_ncx*d_ncy;
			int skmm = skm-d_ncx*d_ncy;
			setSharedData(currrentVars, s_data, tz-1, ty, tx, BSZ_Z+4, BSZ_Y, stride, skm);
			setSharedData(currrentVars, s_data, tz-2, ty, tx, BSZ_Z+4, BSZ_Y, stride, skmm);
		}
		if(threadIdx.z==blockDim.z-1) {
			int skp = s+d_ncx*d_ncy;
			int skpp = skp+d_ncx*d_ncy;
			setSharedData(currrentVars, s_data, tz+1, ty, tx, BSZ_Z+4, BSZ_Y, stride, skp);
			setSharedData(currrentVars, s_data, tz+2, ty, tx, BSZ_Z+4, BSZ_Y, stride, skpp);
		}

		__syncthreads();
		//==========================================================================================

		PRECISION e_s = e[s];
		PRECISION result[NUMBER_CONSERVED_VARIABLES], H[NUMBER_CONSERVED_VARIABLES];
		flux2(s_data, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusZ, &Fz, t, ts, stride, e_s);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = - *(H+n);
		}
		flux2(s_data, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusZ, &Fz, t, ts, stride, e_s);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n);
			*(result+n) /= d_dz;
		}
		// source
		PRECISION facZ = 1/d_dz/2;
		int ptr = ts+4*stride;
		PRECISION dnpitt = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facZ;	ptr+=stride;
		PRECISION dnpitx = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facZ;	ptr+=stride;
		PRECISION dnpity = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facZ;	ptr+=stride;
		PRECISION dnpitn = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facZ;	ptr+=stride; ptr+=stride;  ptr+=stride;
		PRECISION dnpixn = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facZ;	ptr+=stride; ptr+=stride;
		PRECISION dnpiyn = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facZ;	ptr+=stride;
		PRECISION dnpinn = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facZ;	ptr+=stride;

		PRECISION ut = u->ut[s];
		PRECISION un = u->un[s];
		PRECISION vn = fdividef(un, ut);
#ifndef PI
		result[0] += dnpitt*vn - dnpitn;
		result[3] += dnpitn*vn - dnpinn;
#else
		PRECISION dnPi = (*(s_data + ptr + 1) - *(s_data + ptr - 1)) *facZ;
		result[0] += dnpitt*vn - dnpitn - vn*dnPi;
		result[3] += dnpitn*vn - dnpinn - dnPi/powf(t,2.0f);
#endif
		result[1] += dnpitx*vn - dnpixn;
		result[2] += dnpity*vn - dnpiyn;

		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) *= d_dt;
		}
		//==========================================================================================

		updatedVars->ttt[s] += result[0];
		updatedVars->ttx[s] += result[1];
		updatedVars->tty[s] += result[2];
		updatedVars->ttn[s] += result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] += result[4];
		updatedVars->pitx[s] += result[5];
		updatedVars->pity[s] += result[6];
		updatedVars->pitn[s] += result[7];
		updatedVars->pixx[s] += result[8];
		updatedVars->pixy[s] += result[9];
		updatedVars->pixn[s] += result[10];
		updatedVars->piyy[s] += result[11];
		updatedVars->piyn[s] += result[12];
		updatedVars->pinn[s] += result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] += result[14];
#endif
	}
}
/**************************************************************************************************************************************************\

/**************************************************************************************************************************************************/
__global__ 
void eulerStepKernelSource_1D(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up
) {
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadID < d_nElements) {
		unsigned int k = threadID / (d_nx * d_ny) + N_GHOST_CELLS_M;
		unsigned int j = (threadID % (d_nx * d_ny)) / d_nx + N_GHOST_CELLS_M;
		unsigned int i = threadID % d_nx + N_GHOST_CELLS_M;
		unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		PRECISION Q[NUMBER_CONSERVED_VARIABLES];
		PRECISION S[NUMBER_CONSERVED_VARIABLES];

		Q[0] = currrentVars->ttt[s];
		Q[1] = currrentVars->ttx[s];
		Q[2] = currrentVars->tty[s];
		Q[3] = currrentVars->ttn[s];
#ifdef PIMUNU
		Q[4] = currrentVars->pitt[s];
		Q[5] = currrentVars->pitx[s];
		Q[6] = currrentVars->pity[s];
		Q[7] = currrentVars->pitn[s];
		Q[8] = currrentVars->pixx[s];
		Q[9] = currrentVars->pixy[s];
		Q[10] = currrentVars->pixn[s];
		Q[11] = currrentVars->piyy[s];
		Q[12] = currrentVars->piyn[s];
		Q[13] = currrentVars->pinn[s];
#endif
#ifdef PI
		Q[14] = currrentVars->Pi[s];
#endif

		loadSourceTerms2(Q, S, u, up->ut[s], up->ux[s], up->uy[s], up->un[s], t, e[s], p, s);

		PRECISION result[NUMBER_CONSERVED_VARIABLES];
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = *(Q+n) + d_dt * ( *(S+n) );
		}

		updatedVars->ttt[s] = result[0];
		updatedVars->ttx[s] = result[1];
		updatedVars->tty[s] = result[2];
		updatedVars->ttn[s] = result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] = result[4];
		updatedVars->pitx[s] = result[5];
		updatedVars->pity[s] = result[6];
		updatedVars->pitn[s] = result[7];
		updatedVars->pixx[s] = result[8];
		updatedVars->pixy[s] = result[9];
		updatedVars->pixn[s] = result[10];
		updatedVars->piyy[s] = result[11];
		updatedVars->piyn[s] = result[12];
		updatedVars->pinn[s] = result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] = result[14];
#endif
	}
}

__global__
void eulerStepKernelX_1D(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const PRECISION * const __restrict__ e
) {
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadID < d_nElements) {
		unsigned int k = threadID / (d_nx * d_ny) + N_GHOST_CELLS_M;
		unsigned int j = (threadID % (d_nx * d_ny)) / d_nx + N_GHOST_CELLS_M;
		unsigned int i = threadID % d_nx + N_GHOST_CELLS_M;
		unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		PRECISION I[5 * NUMBER_CONSERVED_VARIABLES];
		PRECISION H[NUMBER_CONSERVED_VARIABLES];

		// calculate neighbor cell indices;
		int sim = s-1;
		int simm = sim-1;
		int sip = s+1;
		int sipp = sip+1;

		int ptr=0;
		setNeighborCellsJK2(currrentVars->ttt,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttx,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->tty,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
#ifdef PIMUNU
		setNeighborCellsJK2(currrentVars->pitt,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitx,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pity,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixx,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixy,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyy,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pinn,I,s,ptr,simm,sim,sip,sipp); ptr+=5;
#endif
#ifdef PI
		setNeighborCellsJK2(currrentVars->Pi,I,s,ptr,simm,sim,sip,sipp);
#endif

		PRECISION result[NUMBER_CONSERVED_VARIABLES];
		flux(I, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusX, &Fx, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = - *(H+n);
		}
		flux(I, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusX, &Fx, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n);
			*(result+n) /= d_dx;
		}
#ifndef IDEAL
		loadSourceTermsX(I, H, u, s);
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) += *(H+n);
			*(result+n) *= d_dt;
		}
#else
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) *= d_dt;
		}
#endif
		for (unsigned int n = 4; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) *= d_dt;
		}

		updatedVars->ttt[s] += result[0];
		updatedVars->ttx[s] += result[1];
		updatedVars->tty[s] += result[2];
		updatedVars->ttn[s] += result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] += result[4];
		updatedVars->pitx[s] += result[5];
		updatedVars->pity[s] += result[6];
		updatedVars->pitn[s] += result[7];
		updatedVars->pixx[s] += result[8];
		updatedVars->pixy[s] += result[9];
		updatedVars->pixn[s] += result[10];
		updatedVars->piyy[s] += result[11];
		updatedVars->piyn[s] += result[12];
		updatedVars->pinn[s] += result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] += result[14];
#endif
	}
}

__global__
void eulerStepKernelY_1D(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const PRECISION * const __restrict__ e
) {
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadID < d_nElements) {
		unsigned int k = threadID / (d_nx * d_ny) + N_GHOST_CELLS_M;
		unsigned int j = (threadID % (d_nx * d_ny)) / d_nx + N_GHOST_CELLS_M;
		unsigned int i = threadID % d_nx + N_GHOST_CELLS_M;
		unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		PRECISION J[5* NUMBER_CONSERVED_VARIABLES];
		PRECISION H[NUMBER_CONSERVED_VARIABLES];

		// calculate neighbor cell indices;
		int sjm = s-d_ncx;
		int sjmm = sjm-d_ncx;
		int sjp = s+d_ncx;
		int sjpp = sjp+d_ncx;

		int ptr=0;
		setNeighborCellsJK2(currrentVars->ttt,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttx,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->tty,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
#ifdef PIMUNU
		setNeighborCellsJK2(currrentVars->pitt,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitx,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pity,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixx,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixy,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyy,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pinn,J,s,ptr,sjmm,sjm,sjp,sjpp); ptr+=5;
#endif
#ifdef PI
		setNeighborCellsJK2(currrentVars->Pi,J,s,ptr,sjmm,sjm,sjp,sjpp);
#endif

		PRECISION result[NUMBER_CONSERVED_VARIABLES];
		flux(J, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusY, &Fy, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = - *(H+n);
		}
		flux(J, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusY, &Fy, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n);
			*(result+n) /= d_dy;
		}
#ifndef IDEAL
		loadSourceTermsY(J, H, u, s);
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) += *(H+n);
			*(result+n) *= d_dt;
		}
#else
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) *= d_dt;
		}
#endif
		for (unsigned int n = 4; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) *= d_dt;
		}

		updatedVars->ttt[s] += result[0];
		updatedVars->ttx[s] += result[1];
		updatedVars->tty[s] += result[2];
		updatedVars->ttn[s] += result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] += result[4];
		updatedVars->pitx[s] += result[5];
		updatedVars->pity[s] += result[6];
		updatedVars->pitn[s] += result[7];
		updatedVars->pixx[s] += result[8];
		updatedVars->pixy[s] += result[9];
		updatedVars->pixn[s] += result[10];
		updatedVars->piyy[s] += result[11];
		updatedVars->piyn[s] += result[12];
		updatedVars->pinn[s] += result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] += result[14];
#endif
	}
}

__global__
void eulerStepKernelZ_1D(PRECISION t,
const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
const FLUID_VELOCITY * const __restrict__ u, const PRECISION * const __restrict__ e
) {
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadID < d_nElements) {
		unsigned int k = threadID / (d_nx * d_ny) + N_GHOST_CELLS_M;
		unsigned int j = (threadID % (d_nx * d_ny)) / d_nx + N_GHOST_CELLS_M;
		unsigned int i = threadID % d_nx + N_GHOST_CELLS_M;
		unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		PRECISION K[5 * NUMBER_CONSERVED_VARIABLES];
		PRECISION H[NUMBER_CONSERVED_VARIABLES];

		// calculate neighbor cell indices;
		int stride = d_ncx * d_ncy;
		int skm = s-stride;
		int skmm = skm-stride;
		int skp = s+stride;
		int skpp = skp+stride;

		int ptr=0;
		setNeighborCellsJK2(currrentVars->ttt,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttx,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->tty,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->ttn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
#ifdef PIMUNU
		setNeighborCellsJK2(currrentVars->pitt,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitx,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pity,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pitn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixx,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixy,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pixn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyy,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->piyn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
		setNeighborCellsJK2(currrentVars->pinn,K,s,ptr,skmm,skm,skp,skpp); ptr+=5;
#endif
#ifdef PI
		setNeighborCellsJK2(currrentVars->Pi,K,s,ptr,skmm,skm,skp,skpp);
#endif

		PRECISION result[NUMBER_CONSERVED_VARIABLES];
		flux(K, H, &rightHalfCellExtrapolationForward, &leftHalfCellExtrapolationForward, &spectralRadiusZ, &Fz, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) = -*(H+n);
		}
		flux(K, H, &rightHalfCellExtrapolationBackwards, &leftHalfCellExtrapolationBackwards, &spectralRadiusZ, &Fz, t, e[s]);
		for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) += *(H+n);
			*(result+n) /= d_dz;
		}
#ifndef IDEAL
		loadSourceTermsZ(K, H, u, s, t);
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) += *(H+n);
			*(result+n) *= d_dt;
		}
#else
		for (unsigned int n = 0; n < 4; ++n) {
			*(result+n) *= d_dt;
		}
#endif
		for (unsigned int n = 4; n < NUMBER_CONSERVED_VARIABLES; ++n) {
			*(result+n) *= d_dt;
		}

		updatedVars->ttt[s] += result[0];
		updatedVars->ttx[s] += result[1];
		updatedVars->tty[s] += result[2];
		updatedVars->ttn[s] += result[3];
#ifdef PIMUNU
		updatedVars->pitt[s] += result[4];
		updatedVars->pitx[s] += result[5];
		updatedVars->pity[s] += result[6];
		updatedVars->pitn[s] += result[7];
		updatedVars->pixx[s] += result[8];
		updatedVars->pixy[s] += result[9];
		updatedVars->pixn[s] += result[10];
		updatedVars->piyy[s] += result[11];
		updatedVars->piyn[s] += result[12];
		updatedVars->pinn[s] += result[13];
#endif
#ifdef PI
		updatedVars->Pi[s] += result[14];
#endif
	}
}
/**************************************************************************************************************************************************/
