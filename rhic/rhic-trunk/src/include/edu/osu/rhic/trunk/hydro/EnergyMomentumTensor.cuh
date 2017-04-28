/*
 * EnergyMomentumTensor.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef ENERGYMOMENTUMTENSOR_CUH_
#define ENERGYMOMENTUMTENSOR_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__host__ __device__ 
PRECISION energyDensityFromConservedVariables(PRECISION ePrev, PRECISION M0, PRECISION M1, PRECISION M2, PRECISION M3);

__host__ __device__ 
void getInferredVariables(PRECISION t, const PRECISION * const __restrict__ q, PRECISION ePrev,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p, 
PRECISION * const __restrict__ ut, PRECISION * const __restrict__ ux, PRECISION * const __restrict__ uy, PRECISION * const __restrict__ un
);

__global__ 
void setInferredVariablesKernel(const CONSERVED_VARIABLES * const __restrict__ q, 
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p, FLUID_VELOCITY * const __restrict__ u, 
PRECISION t
);

__host__ __device__ 
PRECISION Ttt(PRECISION e, PRECISION p, PRECISION ut, PRECISION pitt);
__host__ __device__ 
PRECISION Ttx(PRECISION e, PRECISION p, PRECISION ut, PRECISION ux, PRECISION pitx);
__host__ __device__ 
PRECISION Tty(PRECISION e, PRECISION p, PRECISION ut, PRECISION uy, PRECISION pity);
__host__ __device__ 
PRECISION Ttn(PRECISION e, PRECISION p, PRECISION ut, PRECISION un, PRECISION pitn);
__host__ __device__ 
PRECISION Txx(PRECISION e, PRECISION p, PRECISION ux, PRECISION pixx);
__host__ __device__ 
PRECISION Txy(PRECISION e, PRECISION p, PRECISION ux, PRECISION uy, PRECISION pixy);
__host__ __device__ 
PRECISION Txn(PRECISION e, PRECISION p, PRECISION ux, PRECISION un, PRECISION pixn);
__host__ __device__ 
PRECISION Tyy(PRECISION e, PRECISION p, PRECISION uy, PRECISION piyy);
__host__ __device__ 
PRECISION Tyn(PRECISION e, PRECISION p, PRECISION uy, PRECISION un, PRECISION piyn);
__host__ __device__ 
PRECISION Tnn(PRECISION e, PRECISION p, PRECISION un, PRECISION pinn, PRECISION t);

#endif /* ENERGYMOMENTUMTENSOR_CUH_ */
