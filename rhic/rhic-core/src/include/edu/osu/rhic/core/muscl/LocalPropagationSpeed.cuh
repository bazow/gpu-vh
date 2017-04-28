/*
 * LocalPropagationSpeed.cuh
 *
 *  Created on: Oct 23, 2015
 *      Author: bazow
 */

#ifndef LOCALPROPAGATIONSPEED_CUH_
#define LOCALPROPAGATIONSPEED_CUH_

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

__device__ 
PRECISION localPropagationSpeed(PRECISION utr, PRECISION uxr, PRECISION uyr, PRECISION unr,
		PRECISION utl, PRECISION uxl, PRECISION uyl, PRECISION unl,
		PRECISION (*spectralRadius)(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un)
);

#endif /* LOCALPROPAGATIONSPEED_CUH_ */
