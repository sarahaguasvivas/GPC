#include "functions.h"

float kronecker_delta(int h, int j){
    if (h==j){
        return 1.0;
    } else return 0.0;
}

float step(float j){
    if (j<0.0){
        return 0.0;
    } else return 1.0;
}
