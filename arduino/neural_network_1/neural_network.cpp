
#include "neural_network.h"
#include <stdlib.h>

void buildLayers1(){

   
        dense_1 = buildDense(&dense_1_W[0], dense_1_b, 4, 5, 0xB);

        dense_2 = buildDense(&dense_2_W[0], dense_2_b, 5, 9, 0xB);
 

}


float * fwdNN1(float* data)
{
  
   
        data = fwdDense(dense_1, data);

        data = fwdDense(dense_2, data);
 

    return data;
}

