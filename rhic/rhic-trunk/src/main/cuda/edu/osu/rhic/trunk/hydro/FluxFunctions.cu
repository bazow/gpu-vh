/*
 * FluxFunctions.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#include <stdlib.h>
#include <stdio.h> // for printf

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/trunk/hydro/FluxFunctions.cuh"
#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
PRECISION Fx(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un) {
	return ux * q / ut;
}

__device__ 
PRECISION Fy(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un) {
	return uy * q / ut;
}

__device__ 
PRECISION Fz(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un) {
	return un * q / ut;
}
