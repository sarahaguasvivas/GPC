import numpy as np
from cost.neural_network_cost import *
from dyno_model.neural_network_predictor import *

class NewtonRaphson(Optimizer):

    def __init__(self, cost : Cost, d_model : NeuralNetworkPredictor):
        self.cost = cost
        self.d_model = d_model
        super().__init__()

    def __fsolve_newton(self, F, J, u0, rtol=1e-10, maxit=8, verbose=False):
        """
        Jed Brown's algebraic solver
        """
        u = u0.copy()
        Fu = F(u)
        norm0 = numpy.linalg.norm(Fu)
        enorm_last = numpy.linalg.norm(u - numpy.array([1,1]))

        for i in range(maxit):
            du = -numpy.linalg.solve(J(u), Fu)
            u += du
            Fu = F(u)
            norm = numpy.linalg.norm(Fu)
            if verbose:
                enorm = numpy.linalg.norm(u - numpy.array([1,1]))
                print('Newton {:d} anorm {:6.2e} rnorm {:6.2e} eratio {:6.2f}'.
                      format(i+1, norm, norm/norm0, enorm/enorm_last**2))
                enorm_last = enorm
            if norm < rtol * norm0:
                break
        return u, i

    def optimize(self, n, del_u, u):
       """ This is taken from fsolve_newton in """
        F = d_model.compute_function(n, del_u, u)
        J = d_model.compute_hessian(del_u, u)
        self.__fsolve_newton(F, J, u, rtol=1e-8, maxit = 8, verbose=True)



