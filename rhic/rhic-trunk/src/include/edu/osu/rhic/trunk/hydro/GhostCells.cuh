/*
 * GhostCells.cuh
 *
 *  Created on: Jul 6, 2016
 *      Author: bazow
 */

#ifndef GHOSTCELLS_CUH_
#define GHOSTCELLS_CUH_

void setGhostCells(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u
);
__global__
void setGhostCellsKernelI(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u
);
__global__
void setGhostCellsKernelJ(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u
);
__global__
void setGhostCellsKernelK(CONSERVED_VARIABLES * const __restrict__ q,
PRECISION * const __restrict__ e, PRECISION * const __restrict__ p,
FLUID_VELOCITY * const __restrict__ u
);

#endif /* GHOSTCELLS_CUH_ */
