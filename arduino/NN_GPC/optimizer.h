#ifndef _OPTIMIZER_H__
#define _OPTIMIZER_H__

#include <math.h>
#define JMAX 20 // maximum number of iterations

/*
 *
 * The optimizer used here is Newton Raphson
 * The implementation here is taken from:
 * Numerical Recipes in C: The Art of Scientific Computing by Press et. al. (1988)
 *
 */

#include <math.h>
#include "nrutil.h"
#define TINY 1.0e-20;



float * matrix_inverse(float * );
float * newton_rasphon(float * );

#endif
