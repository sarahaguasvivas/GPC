
#include "neural_network.h"
#include <stdlib.h>

void buildLayers(){

   
        dense_1 = buildDense(&dense_1_W[0], dense_1_b, 12, 5, 0xB);

        dense_2 = buildDense(&dense_2_W[0], dense_2_b, 5, 3, 0xB);
 

}


float * fwdNN(float* data)
{
  
   
        data = fwdDense(dense_1, data);

        data = fwdDense(dense_2, data);
 

    return data;
}

