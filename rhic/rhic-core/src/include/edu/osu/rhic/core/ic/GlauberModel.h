/*
 * GlauberModel.h
 *
 *  Created on: Oct 22, 2015
 *      Author: bazow
 */

#ifndef GLAUBERMODEL_H_
#define GLAUBERMODEL_H_

double woodsSaxonDistribution(double r, double A);
void energyDensityTransverseProfileAA(double * const __restrict__ energyDensityTransverse, 
int nx, int ny, double dx, double dy, void * initCondParams);

#endif /* GLAUBERMODEL_H_ */
