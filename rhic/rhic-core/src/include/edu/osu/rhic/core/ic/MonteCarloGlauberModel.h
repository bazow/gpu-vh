/*
 * MonteCarloGlauberModel.h
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef MONTECARLOGLAUBERMODEL_H_
#define MONTECARLOGLAUBERMODEL_H_

void 
monteCarloGlauberEnergyDensityTransverseProfile(double * const __restrict__ energyDensityTransverse, 
int nx, int ny, double dx, double dy, void * initCondParams
);

#endif /* MONTECARLOGLAUBERMODEL_H_ */
