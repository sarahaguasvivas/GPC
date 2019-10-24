#include "optimizer.h"


/* 
 *  This is an implementation of Newton Raphson Applied to the Neural Networks
 * 
 * Taken from 
 * Numerical Recipes in C: The Art of Scientific Computing by Press et.al. (1988)
 *
 */

float * matrix_inverse(float * );
float * newton_rasphon(float * );
