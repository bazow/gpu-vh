/*
 * FullyDiscreteKurganovTadmorScheme.cu
 *
 *  Created on: Oct 23, 2015
 *      Author: bazow
 */

#include <stdlib.h>
#include <stdio.h> // for printf

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/trunk/hydro/GhostCells.cuh"
#include "edu/osu/rhic/harness/hydro/HydroParameters.h"
#include "edu/osu/rhic/harness/ic/InitialConditionParameters.h"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"
#include "edu/osu/rhic/trunk/hydro/FullyDiscreteKurganovTadmorScheme.cuh"
#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/trunk/hydro/RegulateDissipativeCurrents.cuh"
#include "edu/osu/rhic/trunk/hydro/EulerStep.cuh"
#include "edu/osu/rhic/trunk/hydro/HydrodynamicValidity.cuh"

// Parameters put in constant memory
__constant__ int d_nx,d_ny,d_nz,d_ncx,d_ncy,d_ncz,d_nElements,d_nCompElements;
__constant__ PRECISION d_dt,d_dx,d_dy,d_dz,d_etabar;

// One-dimension kernel launch parameters
int gridSizeConvexComb, blockSizeConvexComb;
int gridSizeGhostI, blockSizeGhostI;
int gridSizeGhostJ, blockSizeGhostJ;
int gridSizeGhostK, blockSizeGhostK;
int gridSizeInferredVars, blockSizeInferredVars;
int gridSizeReg, blockSizeReg;

//===========================================
// Number of threads to launch for 3D fused kernels
dim3 BF = dim3(16,16,2);
dim3 GF;
//===========================================

//===========================================
// Number of threads to launch for 1D fused kernels
int grid_fused_1D,block_fused_1D;
//===========================================

//===========================================
// Number of threads to launch for 3D kernels
dim3 grid,block,grid_X,block_X,grid_Y,block_Y,grid_Z,block_Z;
//===========================================

//===========================================
// Number of threads to launch for 3D kernels using shared memory
dim3 BSX,GSX,BSY,GSY,BSZ,GSZ;
//===========================================

//===========================================
// Number of threads to launch for 1D kernels 
int grid_1D,block_1D,gridX_1D,blockX_1D,gridY_1D,blockY_1D,gridZ_1D,blockZ_1D;
//===========================================

void initializeCUDALaunchParameters(void * latticeParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;

	int minGridSizeConvexComb,minGridSizeInferredVars,minGridSizeGhostI,minGridSizeGhostJ,minGridSizeGhostK;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;

	// set up CUDA Kernel launch parameters	
	int len = nx*ny*nz;
	int len2DI = ny*nz;
	int len2DJ = nx*nz;
	int len2DK = nx*ny;
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeConvexComb, &blockSizeConvexComb, (void*)convexCombinationEulerStepKernel, 0, len);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeInferredVars, &blockSizeInferredVars, (void*)setInferredVariablesKernel, 0, len);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeGhostI, &blockSizeGhostI, (void*)setGhostCellsKernelI, 0, len2DI);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeGhostJ, &blockSizeGhostJ, (void*)setGhostCellsKernelJ, 0, len2DJ);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeGhostK, &blockSizeGhostK, (void*)setGhostCellsKernelK, 0, len2DK);
	gridSizeConvexComb = (len + blockSizeConvexComb - 1) / blockSizeConvexComb;
	gridSizeInferredVars = (len + blockSizeInferredVars - 1) / blockSizeInferredVars;
	gridSizeGhostI = (len2DI + blockSizeGhostI - 1) / blockSizeGhostI;
	gridSizeGhostJ = (len2DJ + blockSizeGhostJ - 1) / blockSizeGhostJ;
	gridSizeGhostK = (len2DK + blockSizeGhostK - 1) / blockSizeGhostK;

	printf("===================================================\n");
	printf("blockSizeConvexComb= %d\n", blockSizeConvexComb);
	printf("blockSizeInferredVars= %d\n", blockSizeInferredVars);
	printf("blockSizeGhostI= %d\n", blockSizeGhostI);
	printf("blockSizeGhostJ= %d\n", blockSizeGhostJ);
	printf("blockSizeGhostK= %d\n", blockSizeGhostK);

	/***************************************************************************************************************/
	// Number of threads to launch for regularization kernel
#ifndef IDEAL
	int minGridSizeReg;
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeReg, &blockSizeReg, (void*)regulateDissipativeCurrents, 0, len);
	gridSizeReg = (len + blockSizeReg - 1)/blockSizeReg;
	printf("blockSizeReg= %d\n", blockSizeReg);
#endif
	/***************************************************************************************************************\

	/***************************************************************************************************************\
	// Number of threads to launch for 3D fused kernels
	int minGridSizeEuler_fused_3D, block_fused_3D;
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_fused_3D, &block_fused_3D, (void*)eulerStepKernel, 0, len);
	printf("blockSizeEuler_fused_3D= %d\n", block_fused_3D);
	GF = dim3((nx + BF.x - 1)/BF.x, (ny + BF.y - 1)/BF.y, (nz + BF.z - 1)/BF.z);	
	/***************************************************************************************************************\

	/***************************************************************************************************************\
	// Number of threads to launch for 1D fused kernels
	int minGridSizeEuler_fused_1D;
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_fused_1D, &block_fused_1D, (void*)eulerStepKernel_1D, 0, len);
	grid_fused_1D = (len + block_fused_1D - 1)/ block_fused_1D;
	printf("blockSizeEuler_fused_1D= %d\n", block_fused_1D);
	/***************************************************************************************************************\

	/***************************************************************************************************************/
	// Number of threads to launch for 3D kernels
	block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);
	grid = dim3((nx + block.x - 1)/ block.x, (ny + block.y - 1)/ block.y, (nz + block.z - 1)/ block.z);	
	// X
	block_X = dim3(BLOCK_DIM_X_X, BLOCK_DIM_X_Y, BLOCK_DIM_X_Z);
	grid_X = dim3((nx + block_X.x - 1)/ block_X.x, (ny + block_X.y - 1)/ block_X.y, (nz + block_X.z - 1)/ block_X.z);	
	// Y
	block_Y = dim3(BLOCK_DIM_Y_X, BLOCK_DIM_Y_Y, BLOCK_DIM_Y_Z);
	grid_Y = dim3((nx + block_Y.x - 1)/ block_Y.x, (ny + block_Y.y - 1)/ block_Y.y, (nz + block_Y.z - 1)/ block_Y.z);	
	// Z
	block_Z = dim3(BLOCK_DIM_Z_X, BLOCK_DIM_Z_Y, BLOCK_DIM_Z_Z);
	grid_Z = dim3((nx + block_Z.x - 1)/ block_Z.x, (ny + block_Z.y - 1)/ block_Z.y, (nz + block_Z.z - 1)/ block_Z.z);	

	// print max potential block size from occupancy
	printf("===================================================\n");
	int minGridSizeEuler_3D,blockSizeEuler_3D;
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_3D, &blockSizeEuler_3D, (void*)eulerStepKernelSource, 0, len);
	printf("blockSizeEulerSource_3D= %d\n", blockSizeEuler_3D);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_3D, &blockSizeEuler_3D, (void*)eulerStepKernelX, 0, len);
	printf("blockSizeEulerX_3D= %d\n", blockSizeEuler_3D);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_3D, &blockSizeEuler_3D, (void*)eulerStepKernelY, 0, len);
	printf("blockSizeEulerY_3D= %d\n", blockSizeEuler_3D);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_3D, &blockSizeEuler_3D, (void*)eulerStepKernelZ, 0, len);
	printf("blockSizeEulerZ_3D= %d\n", blockSizeEuler_3D);
	printf("B = (%d, %d, %d),\tTotal blocks = %d\n", block.x, block.y, block.z, block.x*block.y*block.z);
	printf("BX = (%d, %d, %d),\tTotal blocks = %d\n", block_X.x, block_X.y, block_X.z, block_X.x*block_X.y*block_X.z);
	printf("BY = (%d, %d, %d),\tTotal blocks = %d\n", block_Y.x, block_Y.y, block_Y.z, block_Y.x*block_Y.y*block_Y.z);
	printf("BZ = (%d, %d, %d),\tTotal blocks = %d\n", block_Z.x, block_Z.y, block_Z.z, block_Z.x*block_Z.y*block_Z.z);
	/***************************************************************************************************************\

	/***************************************************************************************************************\
	// Number of threads to launch for 3D kernels using shared memory
	block = dim3(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);
	grid = dim3((nx + block.x - 1)/ block.x, (ny + block.y - 1)/ block.y, (nz + block.z - 1)/ block.z);
	// X
	BSX = dim3(BSX_X, BSX_Y, BSX_Z);
	GSX = dim3((nx + BSX_X - 1)/BSX_X, (ny + BSX_Y - 1)/BSX_Y, (nz + BSX_Z - 1)/BSX_Z);	
	// Y
	BSY = dim3(BSY_X, BSY_Y, BSY_Z);
	GSY = dim3((nx + BSY_X - 1)/BSY_X, (ny + BSY_Y - 1)/BSY_Y, (nz + BSY_Z - 1)/BSY_Z);	
	// Z
	BSZ = dim3(BSZ_X, BSZ_Y, BSZ_Z);
	GSZ = dim3((nx + BSZ_X - 1)/BSZ_X, (ny + BSZ_Y - 1)/BSZ_Y, (nz + BSZ_Z - 1)/BSZ_Z);

	// print max potential block size from occupancy
	printf("===================================================\n");
	int minGridSizeEuler_3D_Shared,blockSizeEuler_3D_Shared;
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_3D_Shared, &blockSizeEuler_3D_Shared, (void*)eulerStepKernelSource, 0, len);
	printf("blockSizeEulerSource= %d\n", blockSizeEuler_3D_Shared);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_3D_Shared, &blockSizeEuler_3D_Shared, (void*)eulerStepKernelSharedX, 0, len);
	printf("blockSizeEulerSharedX= %d\n", blockSizeEuler_3D_Shared);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_3D_Shared, &blockSizeEuler_3D_Shared, (void*)eulerStepKernelSharedY, 0, len);
	printf("blockSizeEulerSharedY= %d\n", blockSizeEuler_3D_Shared);
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_3D_Shared, &blockSizeEuler_3D_Shared, (void*)eulerStepKernelSharedZ, 0, len);
	printf("blockSizeEulerSharedZ= %d\n", blockSizeEuler_3D_Shared);
	printf("B = (%d, %d, %d),\tTotal blocks = %d\n", block.x, block.y, block.z, block.x*block.y*block.z);
	printf("BSX = (%d, %d, %d),\tTotal blocks = %d\n", BSX_X, BSX_Y, BSX_Z, BSX_X*BSX_Y*BSX_Z);
	printf("BSY = (%d, %d, %d),\tTotal blocks = %d\n", BSY_X, BSY_Y, BSY_Z, BSY_X*BSY_Y*BSY_Z);
	printf("BSZ = (%d, %d, %d),\tTotal blocks = %d\n", BSZ_X, BSZ_Y, BSZ_Z, BSZ_X*BSZ_Y*BSZ_Z);
	/***************************************************************************************************************/

	/***************************************************************************************************************/
	// Number of threads to launch for 1D kernels
	int minGridSizeEuler_1D;
	// source
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_1D, &block_1D, (void*)eulerStepKernelSource_1D, 0, len);
	grid_1D = (len + block_1D - 1)/ block_1D;
	// X
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_1D, &blockX_1D, (void*)eulerStepKernelX_1D, 0, len);
	gridX_1D = (len + blockX_1D - 1)/ blockX_1D;
	// Y
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_1D, &blockY_1D, (void*)eulerStepKernelY_1D, 0, len);
	gridY_1D = (len + blockY_1D - 1)/ blockY_1D;
	// Z
	cudaOccupancyMaxPotentialBlockSize(&minGridSizeEuler_1D, &blockZ_1D, (void*)eulerStepKernelZ_1D, 0, len);
	gridZ_1D = (len + blockZ_1D - 1)/ blockZ_1D;

	printf("===================================================\n");
	printf("blockSizeEulerSource_1D= %d\n", block_1D);
	printf("blockSizeEulerX_1D= %d\n", blockX_1D);
	printf("blockSizeEulerY_1D= %d\n", blockY_1D);
	printf("blockSizeEulerZ_1D= %d\n", blockZ_1D);
	/***************************************************************************************************************/
	printf("===================================================\n");
}

void initializeCUDAConstantParameters(void * latticeParams, void * initCondParams, void * hydroParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;
	struct HydroParameters * hydro = (struct HydroParameters *) hydroParams;
	struct InitialConditionParameters * initCond = (struct InitialConditionParameters *) initCondParams;

	int nx,ny,nz,ncx,ncy,ncz,nElements,nCompElements;
	PRECISION dt,dx,dy,dz,etabar;

	nx = lattice->numLatticePointsX;
	ny = lattice->numLatticePointsY;
	nz = lattice->numLatticePointsRapidity;
	ncx = lattice->numComputationalLatticePointsX;
	ncy = lattice->numComputationalLatticePointsY;
	ncz = lattice->numComputationalLatticePointsRapidity;

	dt = (PRECISION)(lattice->latticeSpacingProperTime);
	dx = (PRECISION)(lattice->latticeSpacingX);
	dy = (PRECISION)(lattice->latticeSpacingY);
	dz = (PRECISION)(lattice->latticeSpacingRapidity);

	etabar = (PRECISION)(hydro->shearViscosityToEntropyDensity);

	nCompElements = ncx * ncy * ncz;	
	nElements = nx * ny * nz;	

	cudaMemcpyToSymbol(d_nx, &nx, sizeof(nx), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_ny, &ny, sizeof(ny), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nz, &nz, sizeof(nz), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_ncx, &ncx, sizeof(ncx), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_ncy, &ncy, sizeof(ncy), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_ncz, &ncz, sizeof(ncz), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nElements, &nElements, sizeof(nElements), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_nCompElements, &nCompElements, sizeof(nCompElements), 0, cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_dx, &dx, sizeof(dx), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_dy, &dy, sizeof(dy), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_dz, &dz, sizeof(dz), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_etabar, &etabar, sizeof(etabar), 0, cudaMemcpyHostToDevice);
}
