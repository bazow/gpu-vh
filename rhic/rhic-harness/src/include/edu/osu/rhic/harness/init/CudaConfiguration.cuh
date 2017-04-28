/*
 * CudaConfiguration.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef CUDACONFIGURATION_CUH_
#define CUDACONFIGURATION_CUH_

// Parameters put in constant memory
extern __constant__ int d_nx,d_ny,d_nz,d_ncx,d_ncy,d_ncz,d_nElements,d_nCompElements;
extern __constant__ PRECISION d_dt,d_dx,d_dy,d_dz,d_etabar;

// One-dimension kernel launch parameters
extern int gridSizeConvexComb, blockSizeConvexComb;
extern int gridSizeGhostI, blockSizeGhostI;
extern int gridSizeGhostJ, blockSizeGhostJ;
extern int gridSizeGhostK, blockSizeGhostK;
extern int gridSizeInferredVars, blockSizeInferredVars;
extern int gridSizeReg, blockSizeReg;

//===========================================
// Number of threads to launch for 3D fused kernels
extern dim3 BF, GF;
//===========================================

//===========================================
// Number of threads to launch for 1D fused kernels
extern int grid_fused_1D,block_fused_1D;
//===========================================

//===========================================
// Number of threads to launch for 3D kernels
// source kernel
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16
#define BLOCK_DIM_Z 2
// X derivative kernel
#define BLOCK_DIM_X_X 32
#define BLOCK_DIM_X_Y 8
#define BLOCK_DIM_X_Z 2
// Y derivative kernel
#define BLOCK_DIM_Y_X 32
#define BLOCK_DIM_Y_Y 8
#define BLOCK_DIM_Y_Z 2
// Z derivative kernel
#define BLOCK_DIM_Z_X 32
#define BLOCK_DIM_Z_Y 8
#define BLOCK_DIM_Z_Z 2
extern dim3 grid,block,grid_X,block_X,grid_Y,block_Y,grid_Z,block_Z;
//===========================================

//===========================================
// Number of threads to launch for 3D kernels using shared memory
// X derivative kernel
#define BSX_X 128
#define BSX_Y 3
#define BSX_Z 2
// Y derivative kernel
#define BSY_X 2
#define BSY_Y 128
#define BSY_Z 2
// Z derivative kernel
#define BSZ_X 4
#define BSZ_Y 4
#define BSZ_Z 32
extern dim3 BSX,GSX,BSY,GSY,BSZ,GSZ;
//===========================================

//===========================================
// Number of threads to launch for 1D kernels 
extern int grid_1D,block_1D,gridX_1D,blockX_1D,gridY_1D,blockY_1D,gridZ_1D,blockZ_1D;
//===========================================

void initializeCUDALaunchParameters(void * latticeParams);
void initializeCUDAConstantParameters(void * latticeParams, void * initCondParams, void * hydroParams);

#endif /* CUDACONFIGURATION_CUH_ */
