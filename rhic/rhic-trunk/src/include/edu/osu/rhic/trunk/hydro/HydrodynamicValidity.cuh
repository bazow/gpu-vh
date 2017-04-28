/*
 * HydrodynamicValidity.cuh
 *
 *  Created on: Jul 6, 2016
 *      Author: bazow
 */

#ifndef HYDRODYNAMICVALIDITY_CUH_
#define HYDRODYNAMICVALIDITY_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__global__
void checkValidityKernel(PRECISION t,
const VALIDITY_DOMAIN * const __restrict__ v,
const CONSERVED_VARIABLES * const __restrict__ currrentVars,
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up
);

void checkValidity(PRECISION t,
const VALIDITY_DOMAIN * const __restrict__ v,
const CONSERVED_VARIABLES * const __restrict__ currrentVars,
const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p,
const FLUID_VELOCITY * const __restrict__ u, const FLUID_VELOCITY * const __restrict__ up
);

#endif /* HYDRODYNAMICVALIDITY_CUH_ */
