/*
 * FiniteDifference.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef FINITEDIFFERENCE_CUH_
#define FINITEDIFFERENCE_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
PRECISION finiteDifferenceX(const PRECISION * const var, int s, int stride, PRECISION fac);
__device__ 
PRECISION finiteDifferenceY(const PRECISION * const var, int s, int stride, PRECISION fac);
__device__ 
PRECISION finiteDifferenceZ(const PRECISION * const var, int s, int stride, PRECISION fac);

#endif /* FINITEDIFFERENCE_CUH_ */
