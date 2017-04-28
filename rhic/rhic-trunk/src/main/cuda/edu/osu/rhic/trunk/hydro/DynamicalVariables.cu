/*
 * DynamicalVariables.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"

CONSERVED_VARIABLES *q;
CONSERVED_VARIABLES *d_q, *d_Q, *d_qS;

FLUID_VELOCITY *u;
FLUID_VELOCITY *d_u, *d_up, *d_uS;

PRECISION *e, *p;
PRECISION *d_e, *d_p;
PRECISION *d_ut, *d_ux, *d_uy, *d_un, *d_utp, *d_uxp, *d_uyp, *d_unp;
PRECISION *d_ttt, *d_ttx, *d_tty, *d_ttn, *d_pitt, *d_pitx, *d_pity, *d_pitn, *d_pixx, *d_pixy, *d_pixn, *d_piyy, *d_piyn, *d_pinn, *d_Pi;

VALIDITY_DOMAIN *validityDomain, *d_validityDomain;
PRECISION *d_regulations, *d_knudsenNumberTaupi, *d_knudsenNumberTauPi, *d_inverseReynoldsNumberPimunu, *d_inverseReynoldsNumber2Pimunu,
		*d_inverseReynoldsNumberTilde2Pimunu, *d_inverseReynoldsNumberPi, *d_inverseReynoldsNumber2Pi, *d_inverseReynoldsNumberTilde2Pi;
// for debugging
PRECISION *d_taupi, *d_dxux, *d_dyuy, *d_theta;

__host__ __device__
int columnMajorLinearIndex(int i, int j, int k, int nx, int ny) {
	return i + nx * (j + ny * k);
}

void allocateHostMemory(int len) {
	size_t bytes = sizeof(PRECISION);
	e = (PRECISION *) calloc(len, bytes);
	p = (PRECISION *) calloc(len, bytes);

	u = (FLUID_VELOCITY *) calloc(1, sizeof(FLUID_VELOCITY));
	u->ut = (PRECISION *) calloc(len, bytes);
	u->ux = (PRECISION *) calloc(len, bytes);
	u->uy = (PRECISION *) calloc(len, bytes);
	u->un = (PRECISION *) calloc(len, bytes);

	q = (CONSERVED_VARIABLES *) calloc(1, sizeof(CONSERVED_VARIABLES));
	q->ttt = (PRECISION *) calloc(len, bytes);
	q->ttx = (PRECISION *) calloc(len, bytes);
	q->tty = (PRECISION *) calloc(len, bytes);
	q->ttn = (PRECISION *) calloc(len, bytes);
	// allocate space for \pi^\mu\nu
#ifdef PIMUNU
	q->pitt = (PRECISION *) calloc(len, bytes);
	q->pitx = (PRECISION *) calloc(len, bytes);
	q->pity = (PRECISION *) calloc(len, bytes);
	q->pitn = (PRECISION *) calloc(len, bytes);
	q->pixx = (PRECISION *) calloc(len, bytes);
	q->pixy = (PRECISION *) calloc(len, bytes);
	q->pixn = (PRECISION *) calloc(len, bytes);
	q->piyy = (PRECISION *) calloc(len, bytes);
	q->piyn = (PRECISION *) calloc(len, bytes);
	q->pinn = (PRECISION *) calloc(len, bytes);
#endif
	// allocate space for \Pi
#ifdef PI
	q->Pi = (PRECISION *) calloc(len, bytes);
#endif

	validityDomain = (VALIDITY_DOMAIN *) calloc(1, sizeof(VALIDITY_DOMAIN));
	validityDomain->regulations = (PRECISION *) calloc(len, bytes);
	validityDomain->knudsenNumberTaupi = (PRECISION *) calloc(len, bytes);
	validityDomain->knudsenNumberTauPi = (PRECISION *) calloc(len, bytes);
	validityDomain->inverseReynoldsNumberPimunu = (PRECISION *) calloc(len, bytes);
	validityDomain->inverseReynoldsNumber2Pimunu = (PRECISION *) calloc(len, bytes);
	validityDomain->inverseReynoldsNumberTilde2Pimunu = (PRECISION *) calloc(len, bytes);
	validityDomain->inverseReynoldsNumberPi = (PRECISION *) calloc(len, bytes);
	validityDomain->inverseReynoldsNumber2Pi = (PRECISION *) calloc(len, bytes);
	validityDomain->inverseReynoldsNumberTilde2Pi = (PRECISION *) calloc(len, bytes);
	for(int s=0; s<len; ++s) validityDomain->regulations[s] = (PRECISION) 1.0;
	validityDomain->taupi = (PRECISION *) calloc(len, bytes);
	validityDomain->dxux = (PRECISION *) calloc(len, bytes);
	validityDomain->dyuy = (PRECISION *) calloc(len, bytes);
	validityDomain->theta = (PRECISION *) calloc(len, bytes);
}

void allocateIntermidateFluidVelocityDeviceMemory(FLUID_VELOCITY *d_u, size_t size2) {
	PRECISION *d_ut, *d_ux, *d_uy, *d_un;
	cudaMalloc((void **) &d_ut, size2);
	cudaMalloc((void **) &d_ux, size2);
	cudaMalloc((void **) &d_uy, size2);
	cudaMalloc((void **) &d_un, size2);

	cudaMemcpy(&(d_u->ut), &d_ut, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->ux), &d_ux, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->uy), &d_uy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->un), &d_un, sizeof(PRECISION*), cudaMemcpyHostToDevice);
}

void allocateIntermidateConservedVarDeviceMemory(CONSERVED_VARIABLES *d_q, size_t bytes) {
	//=======================================================
	// Conserved variables
	//=======================================================
	PRECISION *d_ttt, *d_ttx, *d_tty, *d_ttn, *d_pitt, *d_pitx, *d_pity, *d_pitn, *d_pixx, *d_pixy, *d_pixn, *d_piyy, *d_piyn, *d_pinn, *d_Pi;
	cudaMalloc((void **) &d_ttt, bytes);
	cudaMalloc((void **) &d_ttx, bytes);
	cudaMalloc((void **) &d_tty, bytes);
	cudaMalloc((void **) &d_ttn, bytes);

	cudaMemcpy(&(d_q->ttt), &d_ttt, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->ttx), &d_ttx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->tty), &d_tty, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->ttn), &d_ttn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	// allocate space for \pi^\mu\nu
#ifdef PIMUNU
	cudaMalloc((void **) &d_pitt, bytes);
	cudaMalloc((void **) &d_pitx, bytes);
	cudaMalloc((void **) &d_pity, bytes);
	cudaMalloc((void **) &d_pitn, bytes);
	cudaMalloc((void **) &d_pixx, bytes);
	cudaMalloc((void **) &d_pixy, bytes);
	cudaMalloc((void **) &d_pixn, bytes);
	cudaMalloc((void **) &d_piyy, bytes);
	cudaMalloc((void **) &d_piyn, bytes);
	cudaMalloc((void **) &d_pinn, bytes);

	cudaMemcpy(&(d_q->pitt), &d_pitt, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pitx), &d_pitx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pity), &d_pity, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pitn), &d_pitn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixx), &d_pixx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixy), &d_pixy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixn), &d_pixn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->piyy), &d_piyy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->piyn), &d_piyn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pinn), &d_pinn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
#endif
	// allocate space for \Pi
#ifdef PI
	cudaMalloc((void **) &d_Pi, bytes);

	cudaMemcpy(&(d_q->Pi), &d_Pi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
#endif
}

void allocateDeviceMemory(size_t bytes) {
	//=======================================================
	// Energy density and pressure
	//=======================================================	
	cudaMalloc((void **) &d_e, bytes);
	cudaMalloc((void **) &d_p, bytes);

	//=======================================================
	// Fluid velocity
	//=======================================================				
	cudaMalloc((void **) &d_ut, bytes);
	cudaMalloc((void **) &d_ux, bytes);
	cudaMalloc((void **) &d_uy, bytes);
	cudaMalloc((void **) &d_un, bytes);
	cudaMalloc((void **) &d_utp, bytes);
	cudaMalloc((void **) &d_uxp, bytes);
	cudaMalloc((void **) &d_uyp, bytes);
	cudaMalloc((void **) &d_unp, bytes);

	cudaMalloc((void**) &d_u, sizeof(FLUID_VELOCITY));
	cudaMalloc((void**) &d_up, sizeof(FLUID_VELOCITY));
	cudaMalloc((void**) &d_uS, sizeof(FLUID_VELOCITY));

	cudaMemcpy(&(d_u->ut), &d_ut, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->ux), &d_ux, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->uy), &d_uy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_u->un), &d_un, sizeof(PRECISION*), cudaMemcpyHostToDevice);

	cudaMemcpy(&(d_up->ut), &d_utp, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_up->ux), &d_uxp, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_up->uy), &d_uyp, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_up->un), &d_unp, sizeof(PRECISION*), cudaMemcpyHostToDevice);

	//=======================================================
	// Conserved variables
	//=======================================================
	cudaMalloc((void **) &d_ttt, bytes);
	cudaMalloc((void **) &d_ttx, bytes);
	cudaMalloc((void **) &d_tty, bytes);
	cudaMalloc((void **) &d_ttn, bytes);

	cudaMalloc((void**) &d_q, sizeof(CONSERVED_VARIABLES));
	cudaMalloc((void**) &d_Q, sizeof(CONSERVED_VARIABLES));
	cudaMalloc((void**) &d_qS, sizeof(CONSERVED_VARIABLES));

	cudaMemcpy(&(d_q->ttt), &d_ttt, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->ttx), &d_ttx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->tty), &d_tty, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->ttn), &d_ttn, sizeof(PRECISION*), cudaMemcpyHostToDevice);

	// allocate space for \pi^\mu\nu
#ifdef PIMUNU
	cudaMalloc((void **) &d_pitt, bytes);
	cudaMalloc((void **) &d_pitx, bytes);
	cudaMalloc((void **) &d_pity, bytes);
	cudaMalloc((void **) &d_pitn, bytes);
	cudaMalloc((void **) &d_pixx, bytes);
	cudaMalloc((void **) &d_pixy, bytes);
	cudaMalloc((void **) &d_pixn, bytes);
	cudaMalloc((void **) &d_piyy, bytes);
	cudaMalloc((void **) &d_piyn, bytes);
	cudaMalloc((void **) &d_pinn, bytes);

	cudaMemcpy(&(d_q->pitt), &d_pitt, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pitx), &d_pitx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pity), &d_pity, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pitn), &d_pitn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixx), &d_pixx, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixy), &d_pixy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pixn), &d_pixn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->piyy), &d_piyy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->piyn), &d_piyn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_q->pinn), &d_pinn, sizeof(PRECISION*), cudaMemcpyHostToDevice);
#endif
	// allocate space for \Pi
#ifdef PI
	cudaMalloc((void **) &d_Pi, bytes);

	cudaMemcpy(&(d_q->Pi), &d_Pi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
#endif

	//=======================================================
	// Intermediate fluid velocity/conserved variables
	//=======================================================
	allocateIntermidateConservedVarDeviceMemory(d_Q, bytes);
	allocateIntermidateConservedVarDeviceMemory(d_qS, bytes);
	allocateIntermidateFluidVelocityDeviceMemory(d_uS, bytes);

	//=======================================================
	// Hydrodynamic validity
	//=======================================================
	cudaMalloc((void **) &d_regulations, bytes);
	cudaMalloc((void **) &d_knudsenNumberTaupi, bytes);
	cudaMalloc((void **) &d_knudsenNumberTauPi, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumberPimunu, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumber2Pimunu, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumberTilde2Pimunu, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumberPi, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumber2Pi, bytes);
	cudaMalloc((void **) &d_inverseReynoldsNumberTilde2Pi, bytes);
	// for debugging purposes
	cudaMalloc((void **) &d_taupi, bytes);
	cudaMalloc((void **) &d_dxux, bytes);
	cudaMalloc((void **) &d_dyuy, bytes);
	cudaMalloc((void **) &d_theta, bytes);

	cudaMalloc((void**) &d_validityDomain, sizeof(VALIDITY_DOMAIN));

	cudaMemcpy(&(d_validityDomain->regulations), &d_regulations, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->knudsenNumberTaupi), &d_knudsenNumberTaupi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->knudsenNumberTauPi), &d_knudsenNumberTauPi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->inverseReynoldsNumberPimunu), &d_inverseReynoldsNumberPimunu, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->inverseReynoldsNumber2Pimunu), &d_inverseReynoldsNumber2Pimunu, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->inverseReynoldsNumberTilde2Pimunu), &d_inverseReynoldsNumberTilde2Pimunu, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->inverseReynoldsNumberPi), &d_inverseReynoldsNumberPi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->inverseReynoldsNumber2Pi), &d_inverseReynoldsNumber2Pi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->inverseReynoldsNumberTilde2Pi), &d_inverseReynoldsNumberTilde2Pi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	// for debugging
	cudaMemcpy(&(d_validityDomain->taupi), &d_taupi, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->dxux), &d_dxux, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->dyuy), &d_dyuy, sizeof(PRECISION*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_validityDomain->theta), &d_theta, sizeof(PRECISION*), cudaMemcpyHostToDevice);
}

void copyHostToDeviceMemory(size_t bytes) {
	cudaMemcpy(d_e, e, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_p, p, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_ut, u->ut, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ux, u->ux, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uy, u->uy, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_un, u->un, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_utp, u->ut, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uxp, u->ux, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_uyp, u->uy, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_unp, u->un, bytes, cudaMemcpyHostToDevice);

	cudaMemcpy(d_ttt, q->ttt, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ttx, q->ttx, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_tty, q->tty, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_ttn, q->ttn, bytes, cudaMemcpyHostToDevice);
	// copy \pi^\mu\nu to device memory
#ifdef PIMUNU
	cudaMemcpy(d_pitt, q->pitt, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pitx, q->pitx, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pity, q->pity, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pitn, q->pitn, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixx, q->pixx, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixy, q->pixy, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pixn, q->pixn, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_piyy, q->piyy, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_piyn, q->piyn, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_pinn, q->pinn, bytes, cudaMemcpyHostToDevice);
#endif
	// copy \Pi to device memory
#ifdef PI
	cudaMemcpy(d_Pi, q->Pi, bytes, cudaMemcpyHostToDevice);
#endif
	cudaMemcpy(d_regulations, validityDomain->regulations, bytes, cudaMemcpyHostToDevice);
}

void copyDeviceToHostMemory(size_t bytes) {
	cudaMemcpy(e, d_e, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(p, d_p, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(u->ut, d_ut, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(u->ux, d_ux, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(u->uy, d_uy, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(u->un, d_un, bytes, cudaMemcpyDeviceToHost);
#ifdef PIMUNU
	cudaMemcpy(q->pitt, d_pitt, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pitx, d_pitx, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pity, d_pity, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pitn, d_pitn, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pixx, d_pixx, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pixy, d_pixy, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pixn, d_pixn, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->piyy, d_piyy, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->piyn, d_piyn, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(q->pinn, d_pinn, bytes, cudaMemcpyDeviceToHost);
#endif
#ifdef PI
	cudaMemcpy(q->Pi, d_Pi, bytes, cudaMemcpyDeviceToHost);
#endif

	cudaMemcpy(validityDomain->regulations, d_regulations, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->knudsenNumberTaupi, d_knudsenNumberTaupi, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->knudsenNumberTauPi, d_knudsenNumberTauPi, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->inverseReynoldsNumberPimunu, d_inverseReynoldsNumberPimunu, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->inverseReynoldsNumber2Pimunu, d_inverseReynoldsNumber2Pimunu, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->inverseReynoldsNumberTilde2Pimunu, d_inverseReynoldsNumberTilde2Pimunu, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->inverseReynoldsNumberPi, d_inverseReynoldsNumberPi, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->inverseReynoldsNumber2Pi, d_inverseReynoldsNumber2Pi, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->inverseReynoldsNumberTilde2Pi, d_inverseReynoldsNumberTilde2Pi, bytes, cudaMemcpyDeviceToHost);
	// for debugging
	cudaMemcpy(validityDomain->taupi, d_taupi, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->dxux, d_dxux, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->dyuy, d_dyuy, bytes, cudaMemcpyDeviceToHost);
	cudaMemcpy(validityDomain->theta, d_theta, bytes, cudaMemcpyDeviceToHost);
}

void setConservedVariables(double t, void * latticeParams) {
	struct LatticeParameters * lattice = (struct LatticeParameters *) latticeParams;

	int nx = lattice->numLatticePointsX;
	int ny = lattice->numLatticePointsY;
	int nz = lattice->numLatticePointsRapidity;
	int ncx = lattice->numComputationalLatticePointsX;
	int ncy = lattice->numComputationalLatticePointsY;

	for (int k = N_GHOST_CELLS_M; k < nz + N_GHOST_CELLS_M; ++k) {
		for (int j = N_GHOST_CELLS_M; j < ny + N_GHOST_CELLS_M; ++j) {
			for (int i = N_GHOST_CELLS_M; i < nx + N_GHOST_CELLS_M; ++i) {
				int s = columnMajorLinearIndex(i, j, k, ncx, ncy);

				PRECISION ux_s = u->ux[s];
				PRECISION uy_s = u->uy[s];
				PRECISION un_s = u->un[s];
				PRECISION ut_s = u->ut[s];
				PRECISION e_s = e[s];
				PRECISION p_s = p[s];

				PRECISION pitt_s = 0;
				PRECISION pitx_s = 0;
				PRECISION pity_s = 0;
				PRECISION pitn_s = 0;
#ifdef PIMUNU
				pitt_s = q->pitt[s];
				pitx_s = q->pitx[s];
				pity_s = q->pity[s];
				pitn_s = q->pitn[s];
#endif
				PRECISION Pi_s = 0;
#ifdef PI
				Pi_s = q->Pi[s];
#endif

				q->ttt[s] = Ttt(e_s, p_s + Pi_s, ut_s, pitt_s);
				q->ttx[s] = Ttx(e_s, p_s + Pi_s, ut_s, ux_s, pitx_s);
				q->tty[s] = Tty(e_s, p_s + Pi_s, ut_s, uy_s, pity_s);
				q->ttn[s] = Ttn(e_s, p_s + Pi_s, ut_s, un_s, pitn_s);
			}
		}
	}
}

void swap(CONSERVED_VARIABLES **arr1, CONSERVED_VARIABLES **arr2) {
	CONSERVED_VARIABLES *tmp = *arr1;
	*arr1 = *arr2;
	*arr2 = tmp;
}

void setCurrentConservedVariables() {
	swap(&d_q, &d_Q);
}

void swapFluidVelocity(FLUID_VELOCITY **arr1, FLUID_VELOCITY **arr2) {
	FLUID_VELOCITY *tmp = *arr1;
	*arr1 = *arr2;
	*arr2 = tmp;
}

void freeHostMemory() {
	free(e);
	free(p);
	free(u->ut);
	free(u->ux);
	free(u->uy);
	free(u->un);
	free(u);

	free(q->ttt);
	free(q->ttx);
	free(q->tty);
	free(q->ttn);
	// free \pi^\mu\nu
#ifdef PIMUNU
	free(q->pitt);
	free(q->pitx);
	free(q->pity);
	free(q->pitn);
	free(q->pixx);
	free(q->pixy);
	free(q->pixn);
	free(q->piyy);
	free(q->piyn);
	free(q->pinn);
#endif
	// free \Pi
#ifdef PI
	free(q->Pi);
#endif
	free(q);
}

void freeDeviceMemory() {
	cudaFree(d_e);
	cudaFree(d_p);

	cudaFree(d_ut);
	cudaFree(d_ux);
	cudaFree(d_uy);
	cudaFree(d_un);
	cudaFree(d_utp);
	cudaFree(d_uxp);
	cudaFree(d_uyp);
	cudaFree(d_unp);

	cudaFree(d_u);
	cudaFree(d_up);
	cudaFree(d_uS);

	cudaFree(d_ttt);
	cudaFree(d_ttx);
	cudaFree(d_tty);
	cudaFree(d_ttn);
	// free \pi^\mu\nu
#ifdef PIMUNU
	cudaFree(d_pitt);
	cudaFree(d_pitx);
	cudaFree(d_pity);
	cudaFree(d_pitn);
	cudaFree(d_pixx);
	cudaFree(d_pixy);
	cudaFree(d_pixn);
	cudaFree(d_piyy);
	cudaFree(d_piyn);
	cudaFree(d_pinn);
#endif
	// free \Pi
#ifdef PI
	cudaFree(d_Pi);
#endif

	cudaFree(d_q);
	cudaFree(d_Q);
	cudaFree(d_qS);
}
