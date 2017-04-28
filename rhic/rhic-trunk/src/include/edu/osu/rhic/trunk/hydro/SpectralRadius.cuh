/*
 * SpectralRadius.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef SPECTRALRADIUS_CUH_
#define SPECTRALRADIUS_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
PRECISION spectralRadiusX(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un);
__device__ 
PRECISION spectralRadiusY(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un);
__device__ 
PRECISION spectralRadiusZ(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un);

#endif /* SPECTRALRADIUS_CUH_ */
