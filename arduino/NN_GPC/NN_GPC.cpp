#include "NN_GPC.h"
#include "constraints.h"
#include "cost.h"
#include "dynamics.h"

/*
 *   
 *        NN_GPC: Neural Network Generalized Predictive Controller
 *   
 *   This file is the equivalent to the test_controller.py code in the Python simulation
 *
 */


float * NN_GPC(float * window, int N1, int N2, int Nu, int K, float ym[], float yn[], float lambda[]) {
  
struct NN_Predictor NN_P;

NN_P = create_dynamic_model(N1, N2, Nu, K, ym, yn, lambda);


  
}
