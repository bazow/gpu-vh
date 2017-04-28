/*
 * SourceTerms.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef SOURCETERMS_CUH_
#define SOURCETERMS_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
void loadSourceTerms(
const PRECISION * const __restrict__ I, const PRECISION * const __restrict__ J, const PRECISION * const __restrict__ K, 
const PRECISION * const __restrict__ Q, PRECISION * const __restrict__ S,
const FLUID_VELOCITY * const __restrict__ u,
PRECISION utp, PRECISION uxp, PRECISION uyp, PRECISION unp,
PRECISION t, PRECISION e, const PRECISION * const __restrict__ pvec,
int s
);
//=================================================================
__device__ 
void loadSourceTermsX(const PRECISION * const __restrict__ I, PRECISION * const __restrict__ S, const FLUID_VELOCITY * const __restrict__ u, int s);
__device__ 
void loadSourceTermsY(const PRECISION * const __restrict__ J, PRECISION * const __restrict__ S, const FLUID_VELOCITY * const __restrict__ u, int s);
__device__ 
void loadSourceTermsZ(const PRECISION * const __restrict__ K, PRECISION * const __restrict__ S, const FLUID_VELOCITY * const __restrict__ u, int s, 
PRECISION t);
__device__ 
void loadSourceTerms2(const PRECISION * const __restrict__ Q, PRECISION * const __restrict__ S, const FLUID_VELOCITY * const __restrict__ u,
PRECISION utp, PRECISION uxp, PRECISION uyp, PRECISION unp,
PRECISION t, PRECISION e, const PRECISION * const __restrict__ pvec,
int s
);

#endif /* SOURCETERMS_CUH_ */
