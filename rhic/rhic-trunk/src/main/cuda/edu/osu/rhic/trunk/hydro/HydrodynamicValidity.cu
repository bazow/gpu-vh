/*
 * HydrodynamicValidity.cu
 *
 *  Created on: Jul 6, 2016
 *      Author: bazow
 */

#include <cuda.h>
#include <cuda_runtime.h>

#include "edu/osu/rhic/trunk/hydro/HydrodynamicValidity.cuh"
#include "edu/osu/rhic/harness/lattice/LatticeParameters.h"
#include "edu/osu/rhic/trunk/hydro/EnergyMomentumTensor.cuh"
#include "edu/osu/rhic/harness/init/CudaConfiguration.cuh"
#include "edu/osu/rhic/trunk/eos/EquationOfState.cuh"
#include "edu/osu/rhic/trunk/hydro/TransportCoefficients.cuh"

__global__
void checkValidityKernel(PRECISION t, const VALIDITY_DOMAIN * const __restrict__ v, const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p, const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up) {
	int i = blockDim.x * blockIdx.x + threadIdx.x + N_GHOST_CELLS_M;
	int j = blockDim.y * blockIdx.y + threadIdx.y + N_GHOST_CELLS_M;
	int k = blockDim.z * blockIdx.z + threadIdx.z + N_GHOST_CELLS_M;

	if ((i < d_ncx - 2) && (j < d_ncy - 2) && (k < d_ncz - 2)) {
		int s = columnMajorLinearIndex(i, j, k, d_ncx, d_ncy);

		PRECISION e_s = e[s];
		PRECISION p_s = p[s];

		PRECISION *utvec = u->ut;
		PRECISION *uxvec = u->ux;
		PRECISION *uyvec = u->uy;
		PRECISION *unvec = u->un;

		PRECISION ut = utvec[s];
		PRECISION ux = uxvec[s];
		PRECISION uy = uyvec[s];
		PRECISION un = unvec[s];

		PRECISION utp = up->ut[s];
		PRECISION uxp = up->ux[s];
		PRECISION uyp = up->uy[s];
		PRECISION unp = up->un[s];

		//=========================================================
		// spatial derivatives of primary variables
		//=========================================================
		PRECISION facX = 1 / d_dx / 2;
		PRECISION facY = 1 / d_dy / 2;
		PRECISION facZ = 1 / d_dz / 2;
		// dx of u^{\mu} components
		PRECISION dxut = (*(utvec + s + 1) - *(utvec + s - 1)) * facX;
		PRECISION dxux = (*(uxvec + s + 1) - *(uxvec + s - 1)) * facX;
		PRECISION dxuy = (*(uyvec + s + 1) - *(uyvec + s - 1)) * facX;
		PRECISION dxun = (*(unvec + s + 1) - *(unvec + s - 1)) * facX;
		// dy of u^{\mu} components
		PRECISION dyut = (*(utvec + s + d_ncx) - *(utvec + s - d_ncx)) * facY;
		PRECISION dyux = (*(uxvec + s + d_ncx) - *(uxvec + s - d_ncx)) * facY;
		PRECISION dyuy = (*(uyvec + s + d_ncx) - *(uyvec + s - d_ncx)) * facY;
		PRECISION dyun = (*(unvec + s + d_ncx) - *(unvec + s - d_ncx)) * facY;
		// dn of u^{\mu} components
		int stride = d_ncx * d_ncy;
		PRECISION dnut = (*(utvec + s + stride) - *(utvec + s - stride)) * facZ;
		PRECISION dnux = (*(uxvec + s + stride) - *(uxvec + s - stride)) * facZ;
		PRECISION dnuy = (*(uyvec + s + stride) - *(uyvec + s - stride)) * facZ;
		PRECISION dnun = (*(unvec + s + stride) - *(unvec + s - stride)) * facZ;

		PRECISION ut2 = ut * ut;
		PRECISION un2 = un * un;
		PRECISION t2 = t * t;
		PRECISION t3 = t * t2;

		// transport coefficients
		PRECISION T = effectiveTemperature(e_s);

		PRECISION cs2 = speedOfSoundSquared(e_s);
		PRECISION a = 0.333333f - cs2;
		PRECISION a2 = a * a;
		PRECISION lambda_Pipi = 1.6f * a;
		PRECISION zetabar = bulkViscosityToEntropyDensity(T);
		PRECISION tauPiInv = 15 * a2 * fdividef(T, zetabar);

		// time derivatives of u
		PRECISION dtut = (ut - utp) / d_dt;
		PRECISION dtux = (ux - uxp) / d_dt;
		PRECISION dtuy = (uy - uyp) / d_dt;
		PRECISION dtun = (un - unp) / d_dt;

		// Covariant derivatives
		PRECISION Dut = ut * dtut + ux * dxut + uy * dyut + un * dnut + t * un * un;
		PRECISION dut = Dut - t * un * un;
		PRECISION dux = ut * dtux + ux * dxux + uy * dyux + un * dnux;
		PRECISION duy = ut * dtuy + ux * dxuy + uy * dyuy + un * dnuy;
		PRECISION dun = ut * dtun + ux * dxun + uy * dyun + un * dnun;
		PRECISION Dun = -t2 * dun - 2 * t * ut * un;

		// expansion rate
		PRECISION theta = ut / t + dtut + dxux + dyuy + dnun;

		// Velocity shear stress tensor
		PRECISION theta3 = theta / 3;
		PRECISION stt = -t * ut * un2 + (dtut - ut * dut) + (ut2 - 1) * theta3;
		PRECISION stx = -(t * un2 * ux) / 2 + (dtux - dxut) / 2 - (ux * dut + ut * dux) / 2 + ut * ux * theta3;
		PRECISION sty = -(t * un2 * uy) / 2 + (dtuy - dyut) / 2 - (uy * dut + ut * duy) / 2 + ut * uy * theta3;
		PRECISION stn = -un * (2 * ut2 + t2 * un2) / (2 * t) + (dtun - dnut / t2) / 2 - (un * dut + ut * dun) / 2 + ut * un * theta3;
		PRECISION sxx = -(dxux + ux * dux) + (1 + ux * ux) * theta3;
		PRECISION sxy = -(dxuy + dyux) / 2 - (uy * dux + ux * duy) / 2 + ux * uy * theta3;
		PRECISION sxn = -ut * ux * un / t - (dxun + dnux / t2) / 2 - (un * dux + ux * dun) / 2 + ux * un * theta3;
		PRECISION syy = -(dyuy + uy * duy) + (1 + uy * uy) * theta3;
		PRECISION syn = -ut * uy * un / t - (dyun + dnuy / t2) / 2 - (un * duy + uy * dun) / 2 + uy * un * theta3;
		PRECISION snn = -ut * (1 + 2 * t2 * un2) / t3 - dnun / t2 - un * dun + (1 / t2 + un2) * theta3;

		// Vorticity tensor
		PRECISION wtx = (dtux + dxut) / 2 + (ux * dut - ut * dux) / 2 + t * un2 * ux / 2;
		PRECISION wty = (dtuy + dyut) / 2 + (uy * dut - ut * duy) / 2 + t * un2 * uy / 2;
		PRECISION wtn = (t2 * dtun + 2 * t * un + dnut) / 2 + (t2 * un * dut - ut * Dun) + t3 * un * un2 / 2;
		PRECISION wxy = (dyux - dxuy) / 2 + (uy * dux - ux * duy) / 2;
		PRECISION wxn = (dnux - t2 * dxun) / 2 + (t2 * un * dux - ux * Dun) / 2;
		PRECISION wyn = (dnuy - t2 * dyun) / 2 + (t2 * un * duy - uy * Dun) / 2;
		// anti-symmetric vorticity components
		PRECISION wxt = wtx;
		PRECISION wyt = wty;
		PRECISION wnt = wtn / t2;
		PRECISION wyx = -wxy;
		PRECISION wnx = -wxn / t2;
		PRECISION wny = -wyn / t2;

		PRECISION Pi = 0;
#ifdef PI
		Pi = currrentVars->Pi[s];
#endif

#ifdef PIMUNU
		PRECISION pitt = currrentVars->pitt[s];
		PRECISION pitx = currrentVars->pitx[s];
		PRECISION pity = currrentVars->pity[s];
		PRECISION pitn = currrentVars->pitn[s];
		PRECISION pixx = currrentVars->pixx[s];
		PRECISION pixy = currrentVars->pixy[s];
		PRECISION pixn = currrentVars->pixn[s];
		PRECISION piyy = currrentVars->piyy[s];
		PRECISION piyn = currrentVars->piyn[s];
		PRECISION pinn = currrentVars->pinn[s];

		// I2
		PRECISION I2tt = theta * pitt;
		PRECISION I2tx = theta * pitx;
		PRECISION I2ty = theta * pity;
		PRECISION I2tn = theta * pitn;
		PRECISION I2xx = theta * pixx;
		PRECISION I2xy = theta * pixy;
		PRECISION I2xn = theta * pixn;
		PRECISION I2yy = theta * piyy;
		PRECISION I2yn = theta * piyn;
		PRECISION I2nn = theta * pinn;

		// I3
		PRECISION I3tt = 2 * (pitx * wtx + pity * wty + pitn * wtn);
		PRECISION I3tx = pitt * wxt + pity * wxy + pitn * wxn + pixx * wtx + pixy * wty + pixn * wtn;
		PRECISION I3ty = pitt * wyt + pitx * wyx + pitn * wyn + pixy * wtx + piyy * wty + piyn * wtn;
		PRECISION I3tn = pitt * wnt + pitx * wnx + pity * wny + pixn * wtx + piyn * wty + pinn * wtn;
		PRECISION I3xx = 2 * (pitx * wxt + pixy * wxy + pixn * wxn);
		PRECISION I3xy = pitx * wyt + pity * wxt + pixx * wyx + piyy * wxy + pixn * wyn + piyn * wxn;
		PRECISION I3xn = pitx * wnt + pitn * wxt + pixx * wnx + pixy * wny + piyn * wxy + pinn * wxn;
		PRECISION I3yy = 2 * (pity * wyt + pixy * wyx + piyn * wyn);
		PRECISION I3yn = pity * wnt + pitn * wyt + pixy * wnx + pixn * wyx + piyy * wny + pinn * wyn;
		PRECISION I3nn = 2 * (pitn * wnt + pixn * wnx + piyn * wny);

		// I4
		PRECISION ux2 = ux * ux;
		PRECISION uy2 = uy * uy;
		PRECISION ps = pitt * stt - 2 * pitx * stx - 2 * pity * sty + pixx * sxx + 2 * pixy * sxy + piyy * syy - 2 * pitn * stn * t2 + 2 * pixn * sxn * t2
				+ 2 * piyn * syn * t2 + pinn * snn * t2 * t2;
		PRECISION ps3 = ps / 3;
		PRECISION I4tt = (pitt * stt - pitx * stx - pity * sty - t2 * pitn * stn) - (1 - ut2) * ps3;
		PRECISION I4tx = (pitt * stx + pitx * stt) / 2 - (pitx * sxx + pixx * stx) / 2 - (pity * sxy + pixy * sty) / 2 - t2 * (pitn * sxn + pixn * stn) / 2
				+ (ut * ux) * ps3;
		PRECISION I4ty = (pitt * sty + pity * stt) / 2 - (pitx * sxy + pixy * stx) / 2 - (pity * syy + piyy * sty) / 2 - t2 * (pitn * syn + piyn * stn) / 2
				+ (ut * uy) * ps3;
		PRECISION I4tn = (pitt * stn + pitn * stt) / 2 - (pitx * sxn + pixn * stx) / 2 - (pity * syn + piyn * sty) / 2 - t2 * (pitn * snn + pinn * stn) / 2
				+ (ut * un) * ps3;
		PRECISION I4xx = (pitx * stx - pixx * sxx - pixy * sxy - t2 * pixn * sxn) + (1 + ux2) * ps3;
		PRECISION I4xy = (pitx * sty + pity * stx) / 2 - (pixx * sxy + pixy * sxx) / 2 - (pixy * syy + piyy * sxy) / 2 - t2 * (pixn * syn + piyn * sxn) / 2
				+ (ux * uy) * ps3;
		PRECISION I4xn = (pitx * stn + pitn * stx) / 2 - (pixx * sxn + pixn * sxx) / 2 - (pixy * syn + piyn * sxy) / 2 - t2 * (pixn * snn + pinn * sxn) / 2
				+ (ux * un) * ps3;
		PRECISION I4yy = (pity * sty - pixy * sxy - piyy * syy - t2 * piyn * syn) + (1 + uy2) * ps3;
		PRECISION I4yn = (pity * stn + pitn * sty) / 2 - (pixy * sxn + pixn * sxy) / 2 - (piyy * syn + piyn * syy) / 2 - t2 * (piyn * snn + pinn * syn) / 2
				+ (uy * un) * ps3;
		PRECISION I4nn = (pitn * stn - pixn * sxn - piyn * syn - t2 * pinn * snn) + (1 / t2 + un2) * ps3;

		PRECISION pipi = pitt * pitt - 2 * pitx * pitx - 2 * pity * pity + pixx * pixx + 2 * pixy * pixy + piyy * piyy - 2 * pitn * pitn * t2
				+ 2 * pixn * pixn * t2 + 2 * piyn * piyn * t2 + pinn * pinn * t2 * t2;
		PRECISION ss = stt * stt - 2 * stx * stx - 2 * sty * sty + sxx * sxx + 2 * sxy * sxy + syy * syy - 2 * stn * stn * t2 + 2 * sxn * sxn * t2
				+ 2 * syn * syn * t2 + snn * snn * t2 * t2;

		PRECISION Jtt = delta_pipi * I2tt - I3tt + tau_pipi * I4tt - lambda_piPi * Pi * stt;
		PRECISION Jtx = delta_pipi * I2tx - I3tx + tau_pipi * I4tx - lambda_piPi * Pi * stx;
		PRECISION Jty = delta_pipi * I2ty - I3ty + tau_pipi * I4ty - lambda_piPi * Pi * sty;
		PRECISION Jtn = delta_pipi * I2tn - I3tn + tau_pipi * I4tn - lambda_piPi * Pi * stn;
		PRECISION Jxx = delta_pipi * I2xx - I3xx + tau_pipi * I4xx - lambda_piPi * Pi * sxx;
		PRECISION Jxy = delta_pipi * I2xy - I3xy + tau_pipi * I4xy - lambda_piPi * Pi * sxy;
		PRECISION Jxn = delta_pipi * I2xn - I3xn + tau_pipi * I4xn - lambda_piPi * Pi * sxn;
		PRECISION Jyy = delta_pipi * I2yy - I3yy + tau_pipi * I4yy - lambda_piPi * Pi * syy;
		PRECISION Jyn = delta_pipi * I2yn - I3yn + tau_pipi * I4yn - lambda_piPi * Pi * syn;
		PRECISION Jnn = delta_pipi * I2nn - I3nn + tau_pipi * I4nn - lambda_piPi * Pi * snn;

		PRECISION J = -delta_PiPi * Pi * theta + lambda_Pipi * ps;
		PRECISION JJ = Jtt * Jtt - 2 * Jtx * Jtx - 2 * Jty * Jty + Jxx * Jxx + 2 * Jxy * Jxy + Jyy * Jyy - 2 * Jtn * Jtn * t2 + 2 * Jxn * Jxn * t2
				+ 2 * Jyn * Jyn * t2 + Jnn * Jnn * t2 * t2;

		v->inverseReynoldsNumberPimunu[s] = sqrtf(fabsf(pipi)) / p_s;
		v->inverseReynoldsNumberTilde2Pimunu[s] = T / 2 / d_etabar / (e_s + p_s) * sqrtf(fabsf(JJ / ss));
		v->inverseReynoldsNumberPi[s] = fabsf(Pi) / p_s;
		v->inverseReynoldsNumberTilde2Pi[s] = fabsf(J / zetabar / theta) * T / (e_s + p_s);
		v->knudsenNumberTaupi[s] = 5 * d_etabar * theta / T;
		v->knudsenNumberTauPi[s] = theta / tauPiInv;
#endif
		//================================================================================================
		// FOR DEBUGGING PURPOSES
		//================================================================================================
		v->taupi[s] = 5*d_etabar/T;
		v->dxux[s] = dxux;
		v->dyuy[s] = dyuy;
		v->theta[s] = theta;
	}
}

void checkValidity(PRECISION t, const VALIDITY_DOMAIN * const __restrict__ v, const CONSERVED_VARIABLES * const __restrict__ currrentVars,
		const PRECISION * const __restrict__ e, const PRECISION * const __restrict__ p, const FLUID_VELOCITY * const __restrict__ u,
		const FLUID_VELOCITY * const __restrict__ up) {
	checkValidityKernel<<<grid, block>>>(t, v, currrentVars, e, p, u, up);
}

