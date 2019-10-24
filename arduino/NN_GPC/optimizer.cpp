#include "optimizer.h"


/* 
 *  This is an implementation of Newton Raphson Applied to the Neural Networks
 * 
 * Taken from 
 * Numerical Recipes in C: The Art of Scientific Computing by Press et.al. (1988)
 *
 */

float rtnewt(void (*funcd)(float, float *, float *), float x1, float x2,
        float xacc)
   {
    void nrerror(char error_text[]);
    int j;
    float df,dx,f,rtn;
    rtn=0.5*(x1+x2); Initial guess.
        for (j=1;j<=JMAX;j++) {
            (*funcd)(rtn,&f,&df);
            dx=f/df;
            rtn -= dx;
            if ((x1-rtn)*(rtn-x2) < 0.0)
                nrerror("Jumped out of brackets in rtnewt");
                if (fabs(dx) < xacc) return rtn; //Convergence.
        }
    nrerror("Maximum number of iterations exceeded in rtnewt");
    return 0.0; 
}
