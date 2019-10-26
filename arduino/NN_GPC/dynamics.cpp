#include "dynamics.h"
#include "neural_network.h"
#include "functions.h"


void create_dynamic_model(int N1, int N2, int Nu, int K, float ym[], float yn[], float lambda[]){

  NN_P.N1 = N1;
  NN_P.N2 = N2;
  NN_P.Nu = Nu;
  NN_P.K = K;
  NN_P.previous_second_der = 0.0;

  for (int i=0; i<2; i++){
    NN_P.ym[i] = ym[i];
    NN_P.yn[i] = yn[i];
    NN_P.lambda[i] = lambda[i];
  }
} 

float __phi_prime(float x){
    return 1.0;
}

float __phi_prime_prime(float x){
  return 0.0;
}

 /*
    ---------------------------------------------------------------------
    Soloway, D., and P.J. Haley, “Neural Generalized Predictive Control,”
    Proceedings of the 1996 IEEE International Symposium on Intelligent
    Control, 1996, pp. 277–281.
    Calculating h'th element of the Jacobian
    Calculating m'th and h'th element of the Hessian
    ---------------------------------------------------------------------
 */
 
float __partial_2_fnet_partial_nph_partial_npm(int h, int m, int j){
    /*
     D2^2f_j(net)
     ------------
    Du(n+h)Du(n+m)
    */
  return __phi_prime(0.0)*__partial_2_net_partial_u_nph_partial_npm(h, m, j) + \
                __phi_prime_prime(0.0) * __partial_net_partial_u(h, j) * \
                          __partial_net_partial_u(m, j);
}

float __partial_2_yn_partial_nph_partial_npm(int h, int m, int j){
     /*
         D^2yn
    ---------------
    Du(n+h) Du(n+m)
    */
   // weights in first layer is dense_1_W
   int hid =  dense_1.output_shape[0];
   float sum_output = 0.0;
   for (int i=0; i<hid; i++){
        sum_output+= dense_1_W[j*dense_1[j*hid+i]*__partial_2_fnet_partial_nph_partial_npm(h, m, j);
   }
   NN_P.previous_second_der = sum_output;
   
   return sum_output;
}

float __partial_2_net_partial_u_nph_partial_npm(int, int, int){
        /*
          D^2 net_j
        -------------
        Du(n+h)Du(n+m)
        */
       float sum_output = 0.0;
       int hid = dense_1.output_shape[0];
       for (int i=0; i < min(NN_P.K, NN_P.dd); i++){
            sum_output+= dense_1_W[j*dense_1[j*hid+i + NN_P.nd + 1]*NN_P.previous_second_der * step_(NN_P.K - i -1);
       }
       return sum_output;
}

float __partial_yn_partial_u(int h, int j){
         /*
           D yn
        -----------
         D u(n+h)
        */

        int hid = dense_1.output_shape[0];
        float sum_output = 0.0;
        for (int i=0; i<hid; i++){
            sum_output += dense_1_w[j*hid + i] * __partial_fnet_partial_u(h, j); 
        }
        NN_P.previouos_first_der = sum_output;
        
        return sum_output;
}
float __partial_fnet_partial_u(int h, int j){
        /*
        D f_j(net)
        ---------
         D u(u+h)
       */

       return __phi_prime(0.0)*__partial_net_partial_u(h, j);
}

float __partial_net_partial_u(int, int){
        /*
         D net_j
        ---------
        D u(n+h)
       */
       
        NN_P.nd = weights.shape[1] - 1
        int hid = dense_1.output_shape[0];
        float sum_output = 0.0
        for (int i=0; i<NN_P.nd; i++){
          if ((NN_P.K - NN_P.Nu) < i){
              sum_output+= dense_1_W[j*hid+ i+ 1]*kronecker_delta(NN_P.K - i, h);
          } else{
              sum_output+= dense_1_W[j*hid+ i+ 1]*kronecker_delta(NN_P.Nu, h);
          }
        }

        for (int i=0 ; i< min(NN_P.K,k NN_P.dd); i++){
              sum_output+= dense_1_W[j*hid+ i+ NN_P.nd + 1] * NN_P.previous_first_der * \
                                step_(NN_P.K - i - 1);
        }

        return sum_output;     
}

float __partial_delta_u_partial_u(int, int){
       /*
        D delta u
        ---------
        D u(n+h)
        */

        return kronecker_delta(h, j) - kronecker_delta(h, j-1);
}
float _compute_hessian(float[], float[]){

  float Hessian[NN_P.Nu][NN_P.Nu];

  for (int h = 0; h<NN_P.Nu; h++){
    for (int m ; m<NN_P.Nu; m++){
      
    }
  }

  
}
float _compute_jacobian(float*, float*);
float * Fu(float*, float*);
float * Ju(float*, float*);
float compute_cost(float*, float*);
float * measure(float*);
float * predict(float*);
