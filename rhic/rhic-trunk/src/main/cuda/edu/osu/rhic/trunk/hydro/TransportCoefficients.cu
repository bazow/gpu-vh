/*
 * TransportCoefficients.cu
 *
 *  Created on: Jul 7, 2016
 *      Author: bazow
 */

#include "edu/osu/rhic/trunk/hydro/TransportCoefficients.cuh"
#include "edu/osu/rhic/trunk/eos/EquationOfState.cuh" // for bulk terms

// paramters for the analytic parameterization of the bulk viscosity \zeta/S
#define A_1 -13.77
#define A_2 27.55
#define A_3 13.45

#define LAMBDA_1 0.9
#define LAMBDA_2 0.25
#define LAMBDA_3 0.9
#define LAMBDA_4 0.22

#define SIGMA_1 0.025
#define SIGMA_2 0.13
#define SIGMA_3 0.0025
#define SIGMA_4 0.022

// TODO: Eliminate branching.
__device__ PRECISION bulkViscosityToEntropyDensity(PRECISION T) {
	PRECISION x = T / 1.01355;
	if (x > 1.05)
		return LAMBDA_1 * exp(-(x - 1) / SIGMA_1) + LAMBDA_2 * exp(-(x - 1) / SIGMA_2) + 0.001;
	else if (x < 0.995)
		return LAMBDA_3 * exp((x - 1) / SIGMA_3) + LAMBDA_4 * exp((x - 1) / SIGMA_4) + 0.03;
	else
		return A_1 * x * x + A_2 * x - A_3;
}
