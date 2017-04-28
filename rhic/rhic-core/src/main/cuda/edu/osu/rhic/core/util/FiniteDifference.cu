/*
 * FiniteDifference.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/core/util/FiniteDifference.cuh"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
PRECISION finiteDifferenceX(const PRECISION * const var, int s, int stride, PRECISION fac) {
	return (*(var + s + stride) - *(var + s - stride)) * fac;
}

__device__ 
PRECISION finiteDifferenceY(const PRECISION * const var, int s, int stride, PRECISION fac) {
	return (*(var + s + stride) - *(var + s - stride)) * fac;
}

__device__ 
PRECISION finiteDifferenceZ(const PRECISION * const var, int s, int stride, PRECISION fac) {
	return (*(var + s + stride) - *(var + s - stride)) * fac;
}
