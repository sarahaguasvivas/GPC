#ifndef _COST_H__
#define _COST_H__

#include "constraints.h"

struct Cost{

    int N1, N2, Nu;
    float ym[2];
    float yn[2];
    float lambda[2];
    float cost;
    struct Constraints c;

};






#endif
