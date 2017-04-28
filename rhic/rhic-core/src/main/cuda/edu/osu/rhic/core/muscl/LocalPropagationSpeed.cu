/*
 * LocalPropagationSpeed.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/core/muscl/LocalPropagationSpeed.cuh"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

// maximal local speed at the cell boundaries x_{j\pm 1/2}
__device__ 
PRECISION localPropagationSpeed(PRECISION utr, PRECISION uxr, PRECISION uyr, PRECISION unr,
		PRECISION utl, PRECISION uxl, PRECISION uyl, PRECISION unl,
		PRECISION (*spectralRadius)(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un)
) {
	PRECISION rhoLeftMovingWave = spectralRadius(utl,uxl,uyl,unl);
	PRECISION rhoRightMovingWave = spectralRadius(utr,uxr,uyr,unr);
	PRECISION a = fmaxf(rhoLeftMovingWave, rhoRightMovingWave);
	return a;
}
