#include "dynamics.h"

struct NN_Predictor create_dynamic_model(int N1, int N2, int Nu, int K, float ym[], float yn[], float lambda[]){



return NN_P;
}




float __phi_prime(float);
float __phi_prime_prime(float);
float __partial_2_fnet_partial_nph_partial_npm(int, int, int);
float __partial_2_yn_partial_nph_partial_npm(int, int, int);
float __partial_2_net_partial_u_nph_partial_npm(int, int, int);
float __partial_yn_partial_u(int, int);
float __partial_fnet_partial_u(int, int);
float __partial_net_partial_u(int, int);
float __partial_delta_u_partial_u(int, int);
float _compute_hessian(float*, float*);
float _compute_jacobian(float*, float*);
float * Fu(float*, float*);
float * Ju(float*, float*);
float compute_cost(float*, float*);
float * measure(float*);
float * predict(float*);
