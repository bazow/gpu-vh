/*
 * EquationOfState.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef EQUATIONOFSTATE_CUH_
#define EQUATIONOFSTATE_CUH_

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"

#define CONFORMAL_EOS

// ideal gas of massless quarks and gluons
//#define EOS_FACTOR 15.6269 // Nc=3, Nf=3
#define EOS_FACTOR 13.8997 // Nc=3, Nf=2.5

// Equilibrium pressure
__host__ __device__ 
PRECISION equilibriumPressure(PRECISION e);

// Speed of sound squared
__host__ __device__
PRECISION speedOfSoundSquared(PRECISION e);

// Effective Temperature
__host__ __device__
PRECISION effectiveTemperature(PRECISION e);

// Equilibrium energy density
__host__ __device__ 
PRECISION equilibriumEnergyDensity(PRECISION T);

#endif /* EQUATIONOFSTATE_CUH_ */
