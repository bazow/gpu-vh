/*
 * FluxLimiter.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef FLUXLIMITER_CUH_
#define FLUXLIMITER_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
PRECISION approximateDerivative(PRECISION x, PRECISION y, PRECISION z);

#endif /* FLUXLIMITER_CUH_ */
