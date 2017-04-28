/*
 * GhostCells.cu
 *
 *  Created on: Jul 6, 2016
 *      Author: bazow
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/trunk/hydro/GhostCells.cuh"
#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"

void setGhostCells(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u
) {
	int nstreams = 3;
   cudaStream_t *streams = (cudaStream_t *) malloc(nstreams * sizeof(cudaStream_t));
#pragma unroll 3
   for (int i = 0; i < nstreams; i++) cudaStreamCreate(&(streams[i]));

	setGhostCellsKernelI<<<gridSizeGhostI, blockSizeGhostI, 0, streams[0]>>>(q,e,p,u);
	setGhostCellsKernelJ<<<gridSizeGhostJ, blockSizeGhostJ, 0, streams[1]>>>(q,e,p,u);
	setGhostCellsKernelK<<<gridSizeGhostK, blockSizeGhostK, 0, streams[2]>>>(q,e,p,u);
#pragma unroll 3
	for (int i = 0; i < nstreams; i++) cudaStreamDestroy(streams[i]);
}

__device__
void setGhostCellVars(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u,
int s, int sBC) {
	e[s] = e[sBC];
	p[s] = p[sBC];
	u->ut[s] = u->ut[sBC];
	u->ux[s] = u->ux[sBC];
	u->uy[s] = u->uy[sBC];
	u->un[s] = u->un[sBC];
	q->ttt[s] = q->ttt[sBC];
	q->ttx[s] = q->ttx[sBC];
	q->tty[s] = q->tty[sBC];
	q->ttn[s] = q->ttn[sBC];
	// set \pi^\mu\nu ghost cells if evolved
#ifdef PIMUNU
	q->pitt[s] = q->pitt[sBC];
	q->pitx[s] = q->pitx[sBC];
	q->pity[s] = q->pity[sBC];
	q->pitn[s] = q->pitn[sBC];
	q->pixx[s] = q->pixx[sBC];
	q->pixy[s] = q->pixy[sBC];
	q->pixn[s] = q->pixn[sBC];
	q->piyy[s] = q->piyy[sBC];
	q->piyn[s] = q->piyn[sBC];
	q->pinn[s] = q->pinn[sBC];
#endif
	// set \Pi ghost cells if evolved
#ifdef PI
	q->Pi[s] = q->Pi[sBC];
#endif
}

__global__
void setGhostCellsKernelI(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u
) {
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < d_ny*d_nz) {
		unsigned int j = (id % d_ny) + N_GHOST_CELLS_M;
		unsigned int k = id / d_ny + N_GHOST_CELLS_M;

		int iBC = 2;
#pragma unroll 2
		for (int i = 0; i <= 1; ++i) {
			unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);
			unsigned int sBC = columnMajorLinearIndex(iBC, j, k, d_ncx, d_ncy);
			setGhostCellVars(q,e,p,u,s,sBC);
		}
		iBC = d_nx + 1;
#pragma unroll 2
		for (int i = d_nx + 2; i <= d_nx + 3; ++i) {
			unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);
			unsigned int sBC = columnMajorLinearIndex(iBC, j, k, d_ncx, d_ncy);
			setGhostCellVars(q,e,p,u,s,sBC);
		}
	}
}

__global__
void setGhostCellsKernelJ(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u
) {
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < d_nx*d_nz) {
		unsigned int i = (id % d_nx) + N_GHOST_CELLS_M;
		unsigned int k = id / d_nx + N_GHOST_CELLS_M;

		int jBC = 2;
#pragma unroll 2
		for (int j = 0; j <= 1; ++j) {
			unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);
			unsigned int sBC = columnMajorLinearIndex(i, jBC, k, d_ncx, d_ncy);
			setGhostCellVars(q,e,p,u,s,sBC);
		}
		jBC = d_ny + 1;
#pragma unroll 2
		for (int j = d_ny + 2; j <= d_ny + 3; ++j) {
			unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);
			unsigned int sBC = columnMajorLinearIndex(i, jBC, k, d_ncx, d_ncy);
			setGhostCellVars(q,e,p,u,s,sBC);
		}
	}
}

__global__
void setGhostCellsKernelK(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u
) {
	unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
	if (id < d_nx*d_ny) {
		unsigned int i = (id % d_nx) + N_GHOST_CELLS_M;
		unsigned int j = id / d_nx + N_GHOST_CELLS_M;

		int kBC = 2;
#pragma unroll 2
		for (int k = 0; k <= 1; ++k) {
			unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);
			unsigned int sBC = columnMajorLinearIndex(i, j, kBC, d_ncx, d_ncy);
			setGhostCellVars(q,e,p,u,s,sBC);
		}
		kBC = d_nz + 1;
#pragma unroll 2
		for (int k = d_nz + 2; k <= d_nz + 3; ++k) {
			unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);
			unsigned int sBC = columnMajorLinearIndex(i, j, kBC, d_ncx, d_ncy);
			setGhostCellVars(q,e,p,u,s,sBC);
		}
	}
}

