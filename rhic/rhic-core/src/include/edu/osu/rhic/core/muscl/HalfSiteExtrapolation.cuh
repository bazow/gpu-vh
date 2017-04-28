/*
 * HalfSiteExtrapolation.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef HALFSITEEXTRAPOLATION_CUH_
#define HALFSITEEXTRAPOLATION_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
PRECISION rightHalfCellExtrapolationForward(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp);
__device__ 
PRECISION rightHalfCellExtrapolationBackwards(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp);
__device__ 
PRECISION leftHalfCellExtrapolationForward(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp);
__device__ 
PRECISION leftHalfCellExtrapolationBackwards(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp);

#endif /* HALFSITEEXTRAPOLATION_CUH_ */
