#ifndef __NN_GPC__H_
#define __NN_GPC__H_

/* 
 *                  NN_GPC:
 *      This is like the main GPC code
 *      that was originally implemented in
 *      Python. It contains the right imports
 *      to run our neural network predictive 
 *      controller. 
 * 
 */

float * NN_GPC(float*);
void buildLayers();
float * fwdNN(float*);

#endif
