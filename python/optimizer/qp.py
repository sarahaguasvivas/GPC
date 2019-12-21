import numpy as np
from .optimizer import *
from cost.cost import *
from dyno_model.dynamic_model import *
import cvxopt
from cvxopt import matrix
import scipy.linalg



class QP(Optimizer):
    """
        This implementation does the following:

        minimize (1/2) x^T P x + q^T x
        subject to Gx <= h
                   Ax = b

        I heavily made use of this: https://osqp.org/docs/examples/mpc.html
        Also this code: https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_modeling/mpc_modeling.py
    """
    def __init__(self):
        super().__init__()

    def _inequality_constraints(self, N, nx, nu, xmin, xmax, umin, umax):
        """
            This function shapes the inequality constraints matrices. It
            was taken from
            https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_modeling/mpc_modeling.py
        """
        G = np.zeros((0, (nx+nu) * N))
        h = np.zeros((0, 1))

        umax = np.reshape(umax, (-1, 1))
        umin = np.reshape(umin, (-1, 1))
        xmax = np.reshape(xmax, (-1, 1))
        xmin = np.reshape(xmin, (-1, 1))

        if umax is not None:
            tG = np.hstack([np.eye(N*nu), np.zeros((N*nu, nx*N))])
            th = np.kron(np.ones((N*nu, 1)), umax)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])

        if umin is not None:
            tG = np.hstack([np.eye(N*nu)*-1.0, np.zeros((N*nu, nx*N))])
            new_u = []
            th = np.kron(np.ones((N, 1)), -umin)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])

        if xmax is not None:
            tG = np.hstack([np.zeros((N * nx, nu * N)), np.eye(N * nx)])
            th = np.kron(np.ones((N, 1)), xmax)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])

        if xmin is not None:
            tG = np.hstack([np.zeros((N*nx, nu*N)), np.eye(N*nx)*-1.0])
            th = np.kron(np.ones((N, 1)), -xmin)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])
        return G, h

    def optimize(self, dynamics, state, u0):
        """
            Using: https://osqp.org/docs/examples/mpc.html
                dynamics-> Object of the class DynamicModel
                state -> current state of the system
                P --> quadratic objective
                q --> linear objective
                G --> soft constraints A matrix
                h --> soft constraint b matrix
                A --> hard constraint A matrix
                b --> hard constraint b marix
                u0 is u_optimal in previous step
        """
        Ad, Bd, _, _ = dynamics._get_FGHM(state, u0)

        [nx, nu] = Bd.shape
        H = scipy.linalg.block_diag(np.kron(np.eye(dynamics.N), dynamics.R), \
                np.kron(np.eye(dynamics.N-1), dynamics.Q), np.eye(nx))

        Aeu = np.kron(np.eye(dynamics.N), - Bd)

        Aex = scipy.linalg.block_diag(np.eye((dynamics.N - 1)*nx), np.eye(nx))
        Aex -= np.kron(np.diag([1.0]*(dynamics.N-1), k=-1), Ad)

        Ae = np.hstack((Aeu, Aex))

        be = np.vstack((Ad, np.zeros(((dynamics.N-1)*nx, nx)))).dot(state)

        P = matrix(H)
        q = matrix(np.zeros((dynamics.N * nx + dynamics.N * nu, 1)))
        Ad = matrix(Ae)
        b = matrix(be)

        G, h = self._inequality_constraints(dynamics.N, nx, nu, dynamics.xmin, dynamics.xmax, dynamics.umin, dynamics.umax)

        G = matrix(G)
        h = matrix(h)

        sol = cvxopt.solvers.qp(P, q, G, h, A = Ad, b = b)

        fx = np.array(sol["x"])
        u_optimal = fx[0:N*nu].reshape(N, nu).T
        x = fx[-N*nx:].reshape(N, nx).T
        x = np.hstack((state, x))
        return x, u_optimal




