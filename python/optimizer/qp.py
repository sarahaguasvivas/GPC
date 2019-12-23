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


        I will be using Equation (20) from this paper to get those matrices

            Y = C_k * xi_hat + theta_k * del_u_k
            J = del_u_k.T * Hk del_u_k - del_u_k.T G_k + const

            Hk = theta.T @ Qe @ Theta + Re
            Gk = 2*Theta.T @ Qe @ error_k
            const = error.T @ Qe @ error

            error = y_ref - C_k @ xi_hat
        """

        [Nx, Nu] = len(state), len(u0)

        Nc = dynamics.Nc
        N = dynamics.N
        pred_state = state

        Theta = np.zeros((N, Nc, Nu, Nu)) # matrix theta from Eqn (20)
        Phi = np.zeros((N, 2, Nx))
        x_pred = []
        A_pred = [] # this is A matrices N times
        for _ in range(N):
            F, G, H, M = dynamics._get_FGHM(pred_state, [0.0, 0.0]) # what's the state w/o any moves
            A_pred+=[F.tolist()]
            pred_state= np.add(np.dot(F, pred_state), np.dot(G, [0,0]))
            x_pred+= [pred_state.tolist()]
        A_pred = np.array(A_pred)

        pred_state = state # restart
        B_pred = []

        for _ in range(N):
            F, G, H, M = dynamics._get_FGHM(pred_state, [0.0, 0.0]) # what's the state w/o any moves
            pred_state= np.add(np.dot(F, pred_state), np.dot(G, [0,0]))
            B_pred += [G.tolist()]
        B_pred = np.array(B_pred)

        Ck = H # doesn't change
        Dk = M

        for i in range(N):
            sub_prod = np.eye(Nx) # temp product (neutral element mult)
            for j in range(Nc):
                if i <= j: # sub diagonal
                    for k in range(j, i):
                        sub_prod = sub_prod @ A_pred[k, :, :]
                    Theta[i, j, :, :] = Ck @ sub_prod @ B_pred[i+Nc-1, :, :]

        for i in range(N):
            sub_prod = np.eye(Nx)
            for j in range(i):
                sub_prod = sub_prod @ A_pred[j, :, :]
            Phi[i, :, :] = Ck @ sub_prod

        Theta = Theta[0, 0, :, :]

        Hk = Theta.T @ dynamics.Q @ Theta + dynamics.R

        x_pred = np.array(x_pred)
        target = np.reshape(np.tile(dynamics.ym, N), (-1, 2))

        sub_prod= []
        for i in range(N):
            prod = Phi[i, :, :] @ np.reshape(x_pred[i, :], (-1, 1))
            prod = np.reshape(prod, (1, -1)).tolist()
            sub_prod += prod

        pred = np.reshape(sub_prod, (-1, 2))
        Err = target - np.array(pred)
        dynamics.Cost.cost = np.linalg.norm(Err[0, :])
 #       if dynamics.umax is not None and dynamics.umin is not None:
 #           G, h = self._inequality_constraints(dynamics.N, Nx, Nu, dynamics.xmin, \
 #                               dynamics.xmax, dynamics.umin, dynamics.umax)

        Gk = 2.0 * Theta.T @ dynamics.Q @ Err.T

        P = matrix(Hk)
        q = matrix(np.zeros((Hk.shape[0], 1)))
        G = matrix(Gk.T)
        h = matrix(np.zeros((Gk.shape[1], 1)))

        Ad = None
        b  = None

        if dynamics.umax is not None and dynamics.umin is not None:
            sol = cvxopt.solvers.qp(P, q, G, h, A = Ad, b = b)
        else:
            sol = cvxopt.solvers.qp(P, q, A = Ad, b = b)

        fx = np.array(sol["x"])
        u_optimal = np.reshape(fx[0:N*Nu, :], (Nc, Nu))
        return u_optimal




