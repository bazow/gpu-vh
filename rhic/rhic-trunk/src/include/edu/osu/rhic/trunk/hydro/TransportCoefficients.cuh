/*
 * TransportCoefficients.cuh
 *
 *  Created on: Jul 7, 2016
 *      Author: bazow
 */

#ifndef TRANSPORTCOEFFICIENTS_CUH_
#define TRANSPORTCOEFFICIENTS_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

//
const PRECISION delta_pipi = 1.33333;
const PRECISION tau_pipi = 1.42857;
const PRECISION lambda_piPi = 1.2;
//
const PRECISION delta_PiPi = 0.666667;

__device__ PRECISION bulkViscosityToEntropyDensity(PRECISION T);

#endif /* TRANSPORTCOEFFICIENTS_CUH_ */
