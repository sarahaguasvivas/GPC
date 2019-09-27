import numpy as np
from .optimizer import *
from cost.cost import *
from dyno_model.dynamic_model import *

class NewtonRaphson(Optimizer):

    def __init__(self, cost : Cost, d_model):
        self.cost = cost
        self.d_model = d_model
        super().__init__()

    def __fsolve_newton(self, u0, del_u, rtol=1e-8, maxit=100, verbose=False):
        """
        Jed Brown's algebraic solver
        """
        u = u0.copy()

        du = del_u.copy()

        Fu = self.d_model.Fu(u, du)

        norm0 = np.linalg.norm(Fu)

        enorm_last = np.linalg.norm(u - np.array([1]*len(u)))

        for i in range(maxit):

            Ju = self.d_model.Ju(u, du)

            du = -np.linalg.solve(Ju, Fu)

            u += du

            Fu = self.d_model.Fu(u, du)

            norm = np.linalg.norm(Fu)

            if verbose:
                enorm = np.linalg.norm(u - np.array([1]*len(u)))
                print('Newton {:d} anorm {:6.2e} rnorm {:6.2e} eratio {:6.2f}'.
                      format(i+1, norm, norm/norm0, enorm/enorm_last**2))
                enorm_last = enorm
            if norm < rtol * norm0:
                break
        return u, i, du

    def optimize(self, u, del_u, verbose):
       """ This is taken from fsolve_newton in """
       return self.__fsolve_newton(u, del_u, rtol=1e-4, maxit = 10, verbose=verbose)



