/*
 * DynamicalVariables.cuh
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef DYNAMICALVARIABLES_CUH_
#define DYNAMICALVARIABLES_CUH_

#define NUMBER_CONSERVATION_LAWS 4

//#define PIMUNU 
//#define PI

/*********************************************************/
#ifndef PI
#define NUMBER_PI_COMPONENTS 0
#else
#define NUMBER_PI_COMPONENTS 1
#endif

#ifndef VMU
#define NUMBER_PROPAGATED_VMU_COMPONENTS 0
#else
#define NUMBER_PROPAGATED_VMU_COMPONENTS 4
#endif

#ifndef PIMUNU
#define NUMBER_PROPAGATED_PIMUNU_COMPONENTS 0
#else
#define NUMBER_PROPAGATED_PIMUNU_COMPONENTS 10
#endif

#define NUMBER_DISSIPATIVE_CURRENTS (NUMBER_PI_COMPONENTS+NUMBER_PROPAGATED_VMU_COMPONENTS+NUMBER_PROPAGATED_PIMUNU_COMPONENTS)

#if NUMBER_DISSIPATIVE_CURRENTS==0
#define IDEAL
#endif

#define NUMBER_CONSERVED_VARIABLES (NUMBER_CONSERVATION_LAWS+NUMBER_DISSIPATIVE_CURRENTS)
/*********************************************************/

#define PRECISION float

#include <cuda.h>
#include <cuda_runtime.h>

// Struct containing the conserved variables
typedef struct 
{
	PRECISION *ttt;
	PRECISION *ttx;
	PRECISION *tty;
	PRECISION *ttn;
#ifdef PIMUNU
	PRECISION *pitt;
	PRECISION *pitx;
	PRECISION *pity;
	PRECISION *pitn;
	PRECISION *pixx;
	PRECISION *pixy;
	PRECISION *pixn;
	PRECISION *piyy;
	PRECISION *piyn;
	PRECISION *pinn;
#endif
#ifdef PI
	PRECISION *Pi;
#endif
} CONSERVED_VARIABLES;

// Struct containing components of the fluid velocity
typedef struct 
{
	PRECISION *ut;
	PRECISION *ux;
	PRECISION *uy;
	PRECISION *un;
} FLUID_VELOCITY;

/****************************************************************************/
typedef struct
{
	PRECISION *regulations;
	PRECISION *knudsenNumberTaupi;
	PRECISION *knudsenNumberTauPi;
	PRECISION *inverseReynoldsNumberPimunu;
	PRECISION *inverseReynoldsNumber2Pimunu;
	PRECISION *inverseReynoldsNumberTilde2Pimunu;
	PRECISION *inverseReynoldsNumberPi;
	PRECISION *inverseReynoldsNumber2Pi;
	PRECISION *inverseReynoldsNumberTilde2Pi;
	PRECISION *taupi;
	PRECISION *dxux;
	PRECISION *dyuy;
	PRECISION *theta;
} VALIDITY_DOMAIN;

// To check the validity of the fluid dynamic effective theory
extern VALIDITY_DOMAIN *validityDomain,*d_validityDomain;
extern PRECISION *d_regulations, *d_knudsenNumberTaupi, *d_knudsenNumberTauPi, *d_inverseReynoldsNumberPimunu,*d_inverseReynoldsNumber2Pimunu,*d_inverseReynoldsNumberTilde2Pimunu,
*d_inverseReynoldsNumberPi,*d_inverseReynoldsNumber2Pi,*d_inverseReynoldsNumberTilde2Pi;
// for debugging
extern PRECISION *d_taupi, *d_dxux, *d_dyuy, *d_theta;
/****************************************************************************/

extern CONSERVED_VARIABLES *q;
extern CONSERVED_VARIABLES *d_q,*d_Q,*d_qS;

extern PRECISION *e, *p;
extern PRECISION *d_e, *d_p, *d_ut, *d_ux, *d_uy, *d_un;

extern PRECISION *d_ttt,*d_ttx,*d_tty,*d_ttn,*d_pitt,*d_pitx,*d_pity,*d_pitn,*d_pixx,*d_pixy,*d_pixn,*d_piyy,*d_piyn,*d_pinn,*d_Pi;

extern FLUID_VELOCITY *u;
extern FLUID_VELOCITY *d_u,*d_up,*d_uS;

__host__ __device__ 
int columnMajorLinearIndex(int i, int j, int k, int nx, int ny);

void allocateHostMemory(int len);
void allocateDeviceMemory(size_t bytes);

void copyHostToDeviceMemory(size_t bytes);
void copyDeviceToHostMemory(size_t bytes);

void setConservedVariables(double t, void * latticeParams);
void setCurrentConservedVariables();
void swapFluidVelocity(FLUID_VELOCITY **arr1, FLUID_VELOCITY **arr2);

void freeHostMemory();
void freeDeviceMemory();

#endif /* DYNAMICALVARIABLES_CUH_ */
