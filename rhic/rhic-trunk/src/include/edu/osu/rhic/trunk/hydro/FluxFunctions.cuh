/*
 * FluxFunctions.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef FLUXFUNCTIONS_CUH_
#define FLUXFUNCTIONS_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
PRECISION Fx(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un);
__device__ 
PRECISION Fy(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un);
__device__ 
PRECISION Fz(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un);

#endif /* FLUXFUNCTIONS_CUH_ */
