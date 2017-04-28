#ifndef REGULATEDISSIPATIVECURRENTS_CUH_
#define REGULATEDISSIPATIVECURRENTS_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__global__ 
void regulateDissipativeCurrents(PRECISION t, 
CONSERVED_VARIABLES * const __restrict__ currrentVars, 
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u,
VALIDITY_DOMAIN * const __restrict__ validityDomain
);

#endif /* REGULATEDISSIPATIVECURRENTS_CUH_ */
