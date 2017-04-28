/*
 * SemiDiscreteKurganovTadmorScheme.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef SEMIDISCRETEKURGANOVTADMORSCHEME_CUH_
#define SEMIDISCRETEKURGANOVTADMORSCHEME_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
void flux(const PRECISION * const __restrict__ data, PRECISION * const __restrict__ result,
		PRECISION (* const rightHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const leftHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const spectralRadius)(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION (* const fluxFunction)(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION t, PRECISION ePrev
);

__device__ 
void flux2(const PRECISION * const __restrict__ data, PRECISION * const __restrict__ result,
		PRECISION (* const rightHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const leftHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const spectralRadius)(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION (* const fluxFunction)(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION t, int ptr, int stride, PRECISION ePrev
);
#endif /* SEMIDISCRETEKURGANOVTADMORSCHEME_CUH_ */
