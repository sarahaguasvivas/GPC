#include "optimizer.h"


/* 
 *  This is an implementation of Newton Raphson Applied to the Neural Networks
 * 
 * Taken from 
 * Numerical Recipes in C: The Art of Scientific Computing by Press et.al. (1988)
 *
 */

#define N ...
float **a,**y,d,*col;
int i,j,*indx;
...
ludcmp(a,N,indx,&d); Decompose the matrix just once.
      for(j=1;j<=N;j++) {// Find inverse by columns.
      for(i=1;i<=N;i++) col[i]=0.0;
      col[j]=1.0;
      lubksb(a,N,indx,col);
      for(i=1;i<=N;i++) y[i][j]=col[i];
}

float [] matrix_inverse(float [] ){



  
}
float * newton_rasphon(float * );
