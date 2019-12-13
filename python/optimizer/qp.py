import numpy as np
from cvxopt import matrix
from .optimizer import *
from cost.cost import *
from dyno_model.dynamic_model import *
from control.matlab import *

class QP(Optimizer):
    """
        This implementation does the following:

        minimize (1/2) x^T P x + q^T x
        subject to Gx <= h
                   Ax = b
    """
    def __init__(self):
        super().__init__()

    def __cvxopt_solve_qp(self, P, q, G=None, h= None, A=None, b=None):
        """
            Function taken from: https://scaron.info/blog/quadratic-programming-in-python.html
        """
        P = 0.5*(P + P.T)
        args = [matrix(P), matrix(q)]
        if G is not None:
            args.extend([matrix(G), matrix(h)])
            if A is not None:
                args.extend([matrix(A), matrix(b)])
        sol = cvxopt.solvers.qp(*args)
        if 'optimal' not in sol['status']:
            return None
        return numpy.array(sol['x']).reshape((P.shape[1],))

    def __quadprog_solve_qp(self, P, q, G=None, h=None, A=None, b=None):
        """
            Function taken from: https://scaron.info/blog/quadratic-programming-in-python.html
        """
        qp_G = .5 * (P + P.T)   # make sure P is symmetric
        qp_a = -q
        if A is not None:
            qp_C = -numpy.vstack([A, G]).T
            qp_b = -numpy.hstack([b, h])
            meq = A.shape[0]
        else:  # no equality constraint
            qp_C = -G.T
            qp_b = -h
            meq = 0
        return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]

    def optimize(self, P, q, G, h, A, b):
        """
            P --> other tuning parameter
            q -->
            G -->
            h -->
            A -->
            b -->
        """
        return self.__cvxopt_solve_qp(P, q, G, h, A, b)


    def lqr_optimize(self, A, b, Q, R, N):
        K, S, E = lqr(A, B, Q, R, [N])
        return S



