#ifndef EULERSTEP_CUH_
#define EULERSTEP_CUH_

/****************************************************************************\
__global__
void eulerStepKernel(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const PRECISION * const __restrict__ e,
		const PRECISION * const __restrict__ p,
		const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up);

/****************************************************************************\
__global__
void eulerStepKernel_1D(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const PRECISION * const __restrict__ e,
		const PRECISION * const __restrict__ p,
		const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up);

/****************************************************************************/
__global__
void eulerStepKernelSource(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const PRECISION * const __restrict__ e,
		const PRECISION * const __restrict__ p,
		const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up);
__global__
void eulerStepKernelX(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const FLUID_VELOCITY * const __restrict__ u,
		const PRECISION * const __restrict__ e);
__global__
void eulerStepKernelY(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const FLUID_VELOCITY * const __restrict__ u,
		const PRECISION * const __restrict__ e);
__global__
void eulerStepKernelZ(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const FLUID_VELOCITY * const __restrict__ u,
		const PRECISION * const __restrict__ e);

/****************************************************************************\
__global__
void eulerStepKernelSharedX(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const FLUID_VELOCITY * const __restrict__ u,
		const PRECISION * const __restrict__ e);
__global__
void eulerStepKernelSharedY(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const FLUID_VELOCITY * const __restrict__ u,
		const PRECISION * const __restrict__ e);
__global__
void eulerStepKernelSharedZ(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const FLUID_VELOCITY * const __restrict__ u,
		const PRECISION * const __restrict__ e);

/****************************************************************************/
__global__
void eulerStepKernelSource_1D(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const PRECISION * const __restrict__ e,
		const PRECISION * const __restrict__ p,
		const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up);
__global__
void eulerStepKernelX_1D(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const FLUID_VELOCITY * const __restrict__ u,
		const PRECISION * const __restrict__ e);
__global__
void eulerStepKernelY_1D(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const FLUID_VELOCITY * const __restrict__ u,
		const PRECISION * const __restrict__ e);
__global__
void eulerStepKernelZ_1D(PRECISION t,
		const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		CONSERVED_VARIABLES * const __restrict__ updatedVars,
		const FLUID_VELOCITY * const __restrict__ u,
		const PRECISION * const __restrict__ e);
/****************************************************************************/

#endif /* EULERSTEP_CUH_ */
