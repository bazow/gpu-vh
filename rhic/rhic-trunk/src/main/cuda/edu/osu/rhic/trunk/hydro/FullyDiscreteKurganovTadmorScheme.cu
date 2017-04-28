/*
 * FullyDiscreteKurganovTadmorScheme.cu
 *
 *  Created on: Oct 23, 2015
 *      Author: bazow
 */

#include <stdlib.h>
#include <stdio.h> // for printf

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/trunk/hydro/FullyDiscreteKurganovTadmorScheme.cuh"
#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/trunk/hydro/GhostCells.cuh"
#include "edu/osu/rhic/core/muscl/SemiDiscreteKurganovTadmorScheme.cuh"
#include "edu/osu/rhic/core/muscl/HalfSiteExtrapolation.cuh"
#include "edu/osu//rhic/trunk/hydro/FluxFunctions.cuh"
#include "edu/osu//rhic/trunk/hydro/SpectralRadius.cuh"
#include "edu/osu/rhic/trunk/hydro/SourceTerms.cuh"
#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"
#include "edu/osu/rhic/trunk/hydro/RegulateDissipativeCurrents.cuh"
#include "edu/osu/rhic/trunk/hydro/EulerStep.cuh"
#include "edu/osu/rhic/trunk/hydro/HydrodynamicValidity.cuh"

//#define EULER_STEP_FUSED
//#define EULER_STEP_FUSED_1D
//#define EULER_STEP_SPLIT
//#define EULER_STEP_SMEM
#define EULER_STEP_SPLIT_1D

void eulerStep(PRECISION t, const CONSERVED_VARIABLES * const __restrict__ currrentVars, CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p, const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up) {
#if defined EULER_STEP_FUSED
	eulerStepKernel<<<GF, BF>>>(t, currrentVars, updatedVars, e, p, u, up);
#elif defined EULER_STEP_FUSED_1D
	eulerStepKernel_1D<<<grid_fused_1D, block_fused_1D>>>(t, currrentVars, updatedVars, e, p, u, up);
#elif defined EULER_STEP_SPLIT
	eulerStepKernelSource<<<grid, block>>>(t, currrentVars, updatedVars, e, p, u, up);
	eulerStepKernelX<<<grid_X, block_X>>>(t, currrentVars, updatedVars, u, e);
	eulerStepKernelY<<<grid_Y, block_Y>>>(t, currrentVars, updatedVars, u, e);
	eulerStepKernelZ<<<grid_Z, block_Z>>>(t, currrentVars, updatedVars, u, e);
#elif defined EULER_STEP_SMEM
	eulerStepKernelSource<<<grid, block>>>(t, currrentVars, updatedVars, e, p, u, up);
	eulerStepKernelSharedX<<<GSX, BSX>>>(t, currrentVars, updatedVars, u, e);
	eulerStepKernelSharedY<<<GSY, BSY>>>(t, currrentVars, updatedVars, u, e);
	eulerStepKernelSharedZ<<<GSZ, BSZ>>>(t, currrentVars, updatedVars, u, e);
#elif defined EULER_STEP_SPLIT_1D
	eulerStepKernelSource_1D<<<grid_1D, block_1D>>>(t, currrentVars, updatedVars, e, p, u, up);
	eulerStepKernelX_1D<<<gridX_1D, blockX_1D>>>(t, currrentVars, updatedVars, u, e);
	eulerStepKernelY_1D<<<gridY_1D, blockY_1D>>>(t, currrentVars, updatedVars, u, e);
	eulerStepKernelZ_1D<<<gridZ_1D, blockZ_1D>>>(t, currrentVars, updatedVars, u, e);
#endif
}

__global__
void convexCombinationEulerStepKernel(const CONSERVED_VARIABLES * const __restrict__ q, CONSERVED_VARIABLES * const __restrict__ Q) {
	unsigned int threadID = blockDim.x * blockIdx.x + threadIdx.x;
	if (threadID < d_nElements) {
		unsigned int k = threadID / (d_nx * d_ny) + N_GHOST_CELLS_M;
		unsigned int j = (threadID % (d_nx * d_ny)) / d_nx + N_GHOST_CELLS_M;
		unsigned int i = threadID % d_nx + N_GHOST_CELLS_M;
		unsigned int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		Q->ttt[s] += q->ttt[s];
		Q->ttt[s] /= 2;
		Q->ttx[s] += q->ttx[s];
		Q->ttx[s] /= 2;
		Q->tty[s] += q->tty[s];
		Q->tty[s] /= 2;
		Q->ttn[s] += q->ttn[s];
		Q->ttn[s] /= 2;
#ifdef PIMUNU
		Q->pitt[s] += q->pitt[s];
		Q->pitt[s] /= 2;
		Q->pitx[s] += q->pitx[s];
		Q->pitx[s] /= 2;
		Q->pity[s] += q->pity[s];
		Q->pity[s] /= 2;
		Q->pitn[s] += q->pitn[s];
		Q->pitn[s] /= 2;
		Q->pixx[s] += q->pixx[s];
		Q->pixx[s] /= 2;
		Q->pixy[s] += q->pixy[s];
		Q->pixy[s] /= 2;
		Q->pixn[s] += q->pixn[s];
		Q->pixn[s] /= 2;
		Q->piyy[s] += q->piyy[s];
		Q->piyy[s] /= 2;
		Q->piyn[s] += q->piyn[s];
		Q->piyn[s] /= 2;
		Q->pinn[s] += q->pinn[s];
		Q->pinn[s] /= 2;
#endif
#ifdef PI
		Q->Pi[s] += q->Pi[s];
		Q->Pi[s] /= 2;
#endif
	}
}

#ifndef IDEAL
#define REGULATE_DISSIPATIVE_CURRENTS
#endif
void twoStepRungeKutta(PRECISION t, PRECISION dt, CONSERVED_VARIABLES * __restrict__ d_q, CONSERVED_VARIABLES * __restrict__ d_Q) {
	//===================================================
	// Predicted step
	//===================================================
	eulerStep(t, d_q, d_qS, d_e, d_p, d_u, d_up);

	t += dt;

	setInferredVariablesKernel<<<gridSizeInferredVars, blockSizeInferredVars>>>(d_qS, d_e, d_p, d_uS, t);

#ifdef REGULATE_DISSIPATIVE_CURRENTS
	regulateDissipativeCurrents<<<gridSizeReg, blockSizeReg>>>(t, d_qS, d_e, d_p, d_uS, d_validityDomain);
#endif

	setGhostCells(d_qS, d_e, d_p, d_uS);

	//===================================================
	// Corrected step
	//===================================================
	eulerStep(t, d_qS, d_Q, d_e, d_p, d_uS, d_u);

	convexCombinationEulerStepKernel<<<gridSizeConvexComb, blockSizeConvexComb>>>(d_q, d_Q);

	swapFluidVelocity(&d_up, &d_u);
	setInferredVariablesKernel<<<gridSizeInferredVars, blockSizeInferredVars>>>(d_Q, d_e, d_p, d_u, t);

#ifdef REGULATE_DISSIPATIVE_CURRENTS	
	regulateDissipativeCurrents<<<gridSizeReg, blockSizeReg>>>(t, d_Q, d_e, d_p, d_u, d_validityDomain);
#endif

	setGhostCells(d_Q, d_e, d_p, d_u);

//#ifndef IDEAL
	checkValidity(t, d_validityDomain, d_q, d_e, d_p, d_u, d_up);
//#endif
	cudaDeviceSynchronize();
}
