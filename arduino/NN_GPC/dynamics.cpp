#include "dynamics.h"
#include "neural_network.h"
#include "functions.h"
#include "constraints.h"


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
float [] _compute_hessian(float[] u, float[] del_u){

  float Hessian[NN_P.Nu*NN_P.Nu];
  float sum_output;
  
  for (int h = 0; h<NN_P.Nu; h++){
    for (int m ; m<NN_P.Nu; m++){
        sum_output =0.0;

        for (int j=NN_P.N1; i<NN_P.N2; i++){

            sum_output+= 2.*(__partial_yn_partial_u(h, j)*__partial_yn_partial_u(m, j) - \
                              __partial_2_yn_partial_nph_partial_npm(h, m, j) * \
                                      (NN_P.ym[j] - NN_P.yn[j]));

        }

        for (int j=0; j<NN_P.Nu; j++){
              sum_output+= 2.*(NN_P.lambda[j]*(__partial_delta_u_partial_u(j, h)* \
                                  __partial_delta_u_partial_u(j, m) + del_u[j]));                              
        }

        for (int j=0; j < NN_P.Nu; j++){

              sum_output += kronecker_delta(h, j)*kronecker_delta(m, j) * \
                            (2.*constraints.s / pow(u[j] + constraints.r / 2. - \
                                  constraints.b, 3)) + 2. * constraints.s / pow(constraints.r/2. + 
                                            constraints.b - u[j], 3);
        }

        int idx = m * NN_P.Nu + h;
        Hessian[idx] = sum_output;   
    }
  }
  return Hessian;
}

float [] _compute_jacobian(float[] u, float[] del_u){
    float dJ[NN_P.Nu];
    int sum_output = 0.0;
    for (int h=0; h<NN_P.Nu; h++){
          sum_output=0.0;
          for (int j=NN_P.N1; j<NN_P.N2; j++){
              sum_output += -2.*(NN_P.ym[j] - NN_P.yn[j])*__partial_yn_partial_u(h, j);
          }

          for (int j=0; j<NN_P.Nu; j++){
              sum_output += 2.*NN_P.lambda[j]*del_u[j]* __partial_delta_u_partial_u(j, h);
          }

          for (int j=0; j<NN_P.Nu; j++){
              sum_output += kronecker_delta(h, j) * (-constraints.s/pow(u[j] + constraints.r/2. - constraints.b, 2) + \
                                    constraints.s / pow(constraints.r/2. + constraints.b - u[j], 2));
          }

          dJ[h] = sum_output;
    }
    return dJ;
}

float [] Fu(float [] u, float [] del_u){
    return _compute_jacobian(u, del_u);
}
float * Ju(float [] u, float [] del_u){
    return _compute_hessian(u, del_u);
}
float compute_cost(float [] u, float [] del_u){
      return _compute_costs(u,  del_u); // TODO;
}

float measure(){
    // TODO;
}

float [] predict(float [] u){
  // TODO
      float * output; 
      // TODO: make window have current moves.
      output = fwdNN(window);
}
