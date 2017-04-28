/*
 * HydroPlugin.c
 *
 *  Created on: Oct 23, 2015
 *      Author: bazow
 */

#include <stdlib.h>
#include <stdio.h> // for printf
#include <cmath>

// for timing
#include <ctime>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/harness/hydro/HydroPlugin.h"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/harness/ic/InitialConditionParameters.h"
#include "edu/osu/rhic/harness/hydro/HydroParameters.h"
#include "edu/osu/rhic/harness/io/FileIO.h"
#include "edu/osu/rhic/trunk/ic/InitialConditions.h"
#include "edu/osu/rhic/trunk/hydro/FullyDiscreteKurganovTadmorScheme.cuh"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"
#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/trunk/eos/EquationOfState.cuh"
#include "edu/osu/rhic/trunk/hydro/GhostCells.cuh"
#include "edu/osu/rhic/trunk/hydro/HydrodynamicValidity.cuh"

#define FREQ 10

void outputDynamicalQuantities(double t, const char *outputDir, void * latticeParams) {
	output(e, t, outputDir, "e", latticeParams);
//	/*
	output(p, t, outputDir, "p", latticeParams);
	output(u->ux, t, outputDir, "ux", latticeParams);	
	output(u->uy, t, outputDir, "uy", latticeParams);	
	output(u->un, t, outputDir, "un", latticeParams);
	output(u->ut, t, outputDir, "ut", latticeParams);
	output(q->ttt, t, outputDir, "ttt", latticeParams);	
	output(q->ttn, t, outputDir, "ttn", latticeParams);
#ifdef PIMUNU	
///*
	output(q->pixx, t, outputDir, "pixx", latticeParams);	
	output(q->pixy, t, outputDir, "pixy", latticeParams);	
	output(q->pixn, t, outputDir, "pixn", latticeParams);	
	output(q->piyy, t, outputDir, "piyy", latticeParams);	
	output(q->piyn, t, outputDir, "piyn", latticeParams);	

	output(q->pitt, t, outputDir, "pitt", latticeParams);	
	output(q->pitx, t, outputDir, "pitx", latticeParams);	
	output(q->pity, t, outputDir, "pity", latticeParams);	
	output(q->pitn, t, outputDir, "pitn", latticeParams);
	output(q->pinn, t, outputDir, "pinn", latticeParams);	
#endif
#ifdef PI
	output(q->Pi, t, outputDir, "Pi", latticeParams);	
#endif
///*
	output(validityDomain->regulations, t, outputDir, "regulations", latticeParams);
	output(validityDomain->inverseReynoldsNumberPimunu, t, outputDir, "Rpi", latticeParams);
	output(validityDomain->inverseReynoldsNumberTilde2Pimunu, t, outputDir, "R2pi", latticeParams);
	output(validityDomain->inverseReynoldsNumberPi, t, outputDir, "RPi", latticeParams);
	output(validityDomain->inverseReynoldsNumberTilde2Pi, t, outputDir, "R2Pi", latticeParams);
	output(validityDomain->knudsenNumberTaupi, t, outputDir, "KnTaupi", latticeParams);
	output(validityDomain->knudsenNumberTauPi, t, outputDir, "KnTauPi", latticeParams);
	// for debugging purposes
	output(validityDomain->taupi, t, outputDir, "taupi", latticeParams);
	output(validityDomain->dxux, t, outputDir, "dxux", latticeParams);
	output(validityDomain->dyuy, t, outputDir, "dyuy", latticeParams);
	output(validityDomain->theta, t, outputDir, "theta", latticeParams);
//*/
}

#define CLOCKS_PER_MILLISEC (CLOCKS_PER_SEC / 1000)
class Stopwatch {
private:
	time_t start, end;
public:
	Stopwatch() {
		start = clock();
		end = 0;
	}
	void tic() {
		start = clock();
	}
	void toc() {
		end = clock();
	}
	double elapsedTime() {
		return ((double) (end - start)) / CLOCKS_PER_MILLISEC;
	}
};

void run(void * latticeParams, void * initCondParams, void * hydroParams, const char *rootDirectory, const char *outputDir) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;
	struct HydroParameters * hydro = (struct HydroParameters *) hydroParams;

	/************************************************************************************\
	 * System configuration (C/CUDA)
	/************************************************************************************/
	int nt = lattice->numProperTimePoints;
	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	int ncx = lattice->numComputationalLatticePointsX;
	int ncy = lattice->numComputationalLatticePointsY;
	int ncz = lattice->numComputationalLatticePointsRapidity;
	int nElements = ncx * ncy * ncz;

	double t0 = hydro->initialProperTimePoint;
	double dt = lattice->latticeSpacingProperTime;

	double e0 = initCond->initialEnergyDensity;

	double freezeoutTemperatureGeV = hydro->freezeoutTemperatureGeV;
	const double hbarc = 0.197326938;
	const double freezeoutTemperature = freezeoutTemperatureGeV/hbarc;
//	const double freezeoutEnergyDensity = e0*pow(freezeoutTemperature,4);
	const double freezeoutEnergyDensity = equilibriumEnergyDensity(freezeoutTemperature);
	printf("Grid size = %d x %d x %d\n", nx, ny, nz);
	printf("spatial resolution = (%.3f, %.3f, %.3f)\n", lattice->latticeSpacingX, lattice->latticeSpacingY, lattice->latticeSpacingRapidity);
	printf("freezeout temperature = %.3f [fm^-1] (eF = %.3f [fm^-4])\n", freezeoutTemperature, freezeoutEnergyDensity);

	printf("eta/s = %.6f\n", hydro->shearViscosityToEntropyDensity);

	// Initialize CUDA kernel parameters
	initializeCUDALaunchParameters(latticeParams);
	initializeCUDAConstantParameters(latticeParams, initCondParams, hydroParams);

	// Allocate host and device memory
	size_t bytes = nElements * sizeof(PRECISION);
	allocateHostMemory(nElements);
	allocateDeviceMemory(bytes);

	/************************************************************************************\
	 * Fluid dynamic initialization 
	/************************************************************************************/
	double t = t0;
	// generate initial conditions 
	setInitialConditions(latticeParams, initCondParams, hydroParams, rootDirectory);
	// Calculate conserved quantities
	setConservedVariables(t, latticeParams);
	// copy conserved/inferred variables to GPU memory
	copyHostToDeviceMemory(bytes);
	// impose boundary conditions with ghost cells
	setGhostCells(d_q,d_e,d_p,d_u);
//#ifndef IDEAL
	checkValidity(t, d_validityDomain, d_q, d_e, d_p, d_u, d_up);
//#endif
	/************************************************************************************\
	 * Evolve the system in time
	/************************************************************************************/
	int ictr = (nx % 2 == 0) ? ncx/2 : (ncx-1)/2;
	int jctr = (ny % 2 == 0) ? ncy/2 : (ncy-1)/2;
	int kctr = (nz % 2 == 0) ? ncz/2 : (ncz-1)/2;	
	int sctr = columnMajorLinearIndex(ictr, jctr, kctr, ncx, ncy);

	Stopwatch sw;
	double totalTime = 0;
	int nsteps = 0;

	for (int n = 1; n <= nt+1; ++n) {
		// copy variables back to host and write to disk
		if ((n-1) % FREQ == 0) {
			copyDeviceToHostMemory(bytes);
			printf("n = %d:%d (t = %.3f),\t (e, p) = (%.3f, %.3f) [GeV/fm^3],\t (T = %.3f [GeV]),\t",
				n - 1, nt, t, e[sctr]*hbarc, p[sctr]*hbarc, effectiveTemperature(e[sctr])*hbarc);
			outputDynamicalQuantities(t, outputDir, latticeParams);
			// end hydrodynamic simulation if the temperature is below the freezeout temperature
			if(e[sctr] < freezeoutEnergyDensity) {
				printf("\nReached freezeout temperature at the center.\n");
				break;
			}
		}
		sw.tic();
		twoStepRungeKutta(t, dt, d_q, d_Q);
		sw.toc();
		float elapsedTime = sw.elapsedTime();
		if ((n-1) % FREQ == 0) printf("(Elapsed time/step: %.3f ms)\n", elapsedTime);
		totalTime+=elapsedTime;
		++nsteps;

		setCurrentConservedVariables();

		t = t0 + n * dt;
	}
	printf("Average time/step: %.3f ms\n",totalTime/((double)nsteps));

	/************************************************************************************\
	 * Deallocate host and device memory
	/************************************************************************************/
	freeHostMemory();
	freeDeviceMemory();
	cudaDeviceReset();
}
