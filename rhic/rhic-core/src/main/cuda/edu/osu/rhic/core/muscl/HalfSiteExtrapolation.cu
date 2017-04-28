/*
 * HalfSiteExtrapolation.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/core/muscl/HalfSiteExtrapolation.cuh"
#include "edu/osu/rhic/core/muscl/FluxLimiter.cuh"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
PRECISION rightHalfCellExtrapolationForward(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp) {
	return qp - approximateDerivative(q, qp, qpp)/2;
}
__device__ 
PRECISION rightHalfCellExtrapolationBackwards(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp) {
	return q - approximateDerivative(qm, q, qp)/2;
}
__device__ 
PRECISION leftHalfCellExtrapolationForward(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp) {
	return q + approximateDerivative(qm, q, qp)/2;
}
__device__ 
PRECISION leftHalfCellExtrapolationBackwards(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp) {
	return qm + approximateDerivative(qmm, qm, q)/2;
}
