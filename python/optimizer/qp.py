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

        if umax is not None:
            umax = np.reshape(umax, (-1, 1))
            tG = np.hstack([np.eye(N*nu), np.zeros((N*nu, nx*N))])
            th = np.kron(np.ones((N*nu, 1)), umax)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])

        if umin is not None:
            umin = np.reshape(umin, (-1, 1))
            tG = np.hstack([np.eye(N*nu)*-1.0, np.zeros((N*nu, nx*N))])
            th = np.kron(np.ones((N, 1)), -umin)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])

        if xmax is not None:
            xmax = np.reshape(xmax, (-1, 1))
            tG = np.hstack([np.zeros((N * nx, nu * N)), np.eye(N * nx)])
            th = np.kron(np.ones((N, 1)), xmax)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])

        if xmin is not None:
            xmin = np.reshape(xmin, (-1, 1))
            tG = np.hstack([np.zeros((N*nx, nu*N)), np.eye(N*nx)*-1.0])
            th = np.kron(np.ones((N, 1)), -xmin)
            G = np.vstack([G, tG])
            h = np.vstack([h, th])

        G = np.vstack([G, np.zeros((10, (nx + nu)* N))])
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


        According to https://borrelli.me.berkeley.edu/pdfpub/pub-6.pdf,
        predictions happen as:

            Y = C_k * xi_hat + theta_k * del_u_k
            J = del_u_k.T * Hk del_u_k - del_u_k.T G_k + const

            Hk = theta.T @ Qe @ Theta + Re
            Gk = 2*Theta.T @ Qe @ error_k
            const = error.T @ Qe @ error

            error = y_ref - C_k @ xi_hat
        """
        Ad, Bd, Ck, Dk = dynamics._get_FGHM(state, u0)

        [nx, nu] = Bd.shape

        # Made some modifications to match the required sizes
        #H = scipy.linalg.block_diag(np.kron(np.eye(dynamics.N * nu), dynamics.R), \
        #        np.kron(np.eye((dynamics.N - 1)* nu), dynamics.Q), np.eye(nx - nu))

        Re = dynamics.R * np.eye(2)
        Qe = scipy.linalg.block_diag(dynamics.Q, dynamics.Q, dynamics.Q)
        Hk = Dk.T @ Qe @ Dk + Re

        Aeu = np.kron(np.eye(dynamics.N), - Bd)

        Aex = scipy.linalg.block_diag(np.eye((dynamics.N - 1)*nx), np.eye(nx))
        Aex -= np.kron(np.diag([1.0]*(dynamics.N-1), k=-1), Ad)

        Ae = np.hstack((Aeu, Aex))

        be = np.vstack((Ad, np.zeros(((dynamics.N-1)*nx, \
                        nx)))) @ np.reshape(state, (-1, 1))

        P = matrix(H)

        q = matrix(np.zeros((H.shape[0], 1)))

        Ad = matrix(Ae)
        b = matrix(be)

        if dynamics.umax is not None and dynamics.umin is not None:
            G, h = self._inequality_constraints(dynamics.N, nx, nu, dynamics.xmin, \
                                dynamics.xmax, dynamics.umin, dynamics.umax)

        print(H.shape, (H.shape[0], 1), G.shape, h.shape)

        G = matrix(G)
        h = matrix(h)

        if dynamics.umax is not None and dynamics.umin is not None:
            sol = cvxopt.solvers.qp(P, q, G, h, A = Ad, b = b)
        else:
            sol = cvxopt.solvers.gp(P, q, A = Ad, b = b)

        fx = np.array(sol["x"])
        N = dynamics.N
        u_optimal = fx[0:N*nu].reshape(N, nu).T
        x = fx[-N*nx:].reshape(N, nx).T
        state = np.reshape(state, (1, -1))
        x = np.vstack((state, x.T))
        u1 , u2 = u_optimal
        print("size of u_opt: ", u1.shape, u2.shape)
        return x, u_optimal




