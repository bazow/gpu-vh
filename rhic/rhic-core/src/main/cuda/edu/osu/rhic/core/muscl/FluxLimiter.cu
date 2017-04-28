/*
 * FluxLimiter.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/core/muscl/FluxLimiter.cuh"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

#define THETA 1.8

__device__ 
inline PRECISION sign(PRECISION x) {
	if (x<0) return -1;
	else return 1;
}

__device__ 
inline PRECISION minmod(PRECISION x, PRECISION y) {
	return (sign(x)+sign(y))*fminf(fabsf(x),fabsf(y))/2;
}

__device__ 
PRECISION minmod3(PRECISION x, PRECISION y, PRECISION z) {
   return minmod(x, minmod(y,z));
}

__device__ 
PRECISION approximateDerivative(PRECISION x, PRECISION y, PRECISION z) {
	PRECISION left = THETA * (y - x);
	PRECISION ctr = (z - x) / 2;
	PRECISION right = THETA * (z - y);
	return minmod3(left, ctr, right);
}
