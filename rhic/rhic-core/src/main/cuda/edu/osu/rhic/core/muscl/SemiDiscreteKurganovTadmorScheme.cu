/*
 * SemiDiscreteKurganovTadmorScheme.cu
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */
#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/core/muscl/SemiDiscreteKurganovTadmorScheme.cuh"
#include "edu/osu/rhic/trunk/hydro/DynamicalVariables.cuh"
#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/core/muscl/LocalPropagationSpeed.cuh"

__device__ 
void flux(const PRECISION * const __restrict__ data, PRECISION * const __restrict__ result,
		PRECISION (* const rightHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const leftHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const spectralRadius)(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION (* const fluxFunction)(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION t, PRECISION ePrev
) {
	// left and right cells
	PRECISION qR[NUMBER_CONSERVED_VARIABLES], qL[NUMBER_CONSERVED_VARIABLES];

	// left and right extrapolated values of the conserved variables
	int ptr = 0;
	PRECISION qmm, qm, q, qp, qpp;
	for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
		qmm 	= *(data+ptr);
		qm 	= *(data+ptr+1);
		q 		= *(data+ptr+2);
		qp 	= *(data+ptr+3);
		qpp 	= *(data+ptr+4);
		ptr+=5;
		qR[n]	= rightHalfCellExtrapolation(qmm, qm, q, qp, qpp);
		qL[n]	= leftHalfCellExtrapolation(qmm, qm, q, qp, qpp);
	}

	// left and right extrapolated values of the primary variables
	PRECISION eR,pR,utR,uxR,uyR,unR;
	getInferredVariables(t,qR,ePrev,&eR,&pR,&utR,&uxR,&uyR,&unR);
	PRECISION eL,pL,utL,uxL,uyL,unL;
	getInferredVariables(t,qL,ePrev,&eL,&pL,&utL,&uxL,&uyL,&unL);

	PRECISION a,qR_n,qL_n,FqR,FqL,res;
	a = localPropagationSpeed(utR,uxR,uyR,unR,utL,uxL,uyL,unL,spectralRadius);
	for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
		qR_n = qR[n];
		qL_n = qL[n];
		FqR = fluxFunction(qR_n, utR, uxR, uyR, unR);
		FqL = fluxFunction(qL_n, utL, uxL, uyL, unL);
		res = FqR + FqL - a * (qR_n - qL_n);
		res /= 2;
		result[n] = res; 
	}
}

__device__ 
void flux2(const PRECISION * const __restrict__ data, PRECISION * const __restrict__ result,
		PRECISION (* const rightHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const leftHalfCellExtrapolation)(PRECISION qmm, PRECISION qm, PRECISION q, PRECISION qp, PRECISION qpp),
		PRECISION (* const spectralRadius)(PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION (* const fluxFunction)(PRECISION q, PRECISION ut, PRECISION ux, PRECISION uy, PRECISION un),
		PRECISION t, int ptr, int stride, PRECISION ePrev
) {
	// left and right cells
	PRECISION qR[NUMBER_CONSERVED_VARIABLES], qL[NUMBER_CONSERVED_VARIABLES];

	// left and right extrapolated values of the conserved variables
	for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
		PRECISION q = *(data + ptr);		
		PRECISION qmm 	= *(data + ptr - 2);
		PRECISION qm 	= *(data + ptr - 1);
		PRECISION qp 	= *(data + ptr + 1);
		PRECISION qpp 	= *(data + ptr + 2);
		ptr+=stride;
		qR[n]	= rightHalfCellExtrapolation(qmm, qm, q, qp, qpp);
		qL[n]	= leftHalfCellExtrapolation(qmm, qm, q, qp, qpp);
	}

	// left and right extrapolated values of the primary variables
	PRECISION eR,pR,utR,uxR,uyR,unR;
	getInferredVariables(t,qR,ePrev,&eR,&pR,&utR,&uxR,&uyR,&unR);
	PRECISION eL,pL,utL,uxL,uyL,unL;
	getInferredVariables(t,qL,ePrev,&eL,&pL,&utL,&uxL,&uyL,&unL);

	PRECISION a,qR_n,qL_n,FqR,FqL,res;
	a = localPropagationSpeed(utR,uxR,uyR,unR,utL,uxL,uyL,unL,spectralRadius);
	for (unsigned int n = 0; n < NUMBER_CONSERVED_VARIABLES; ++n) {
		qR_n = qR[n];
		qL_n = qL[n];
		FqR = fluxFunction(qR_n, utR, uxR, uyR, unR);
		FqL = fluxFunction(qL_n, utL, uxL, uyL, unL);
		res = FqR + FqL - a * (qR_n - qL_n);
		res /= 2;
		result[n] = res; 
	}
}
