import numpy as np
import osqp
from .optimizer import *
from cost.cost import *
from dyno_model.dynamic_model import *
from control.matlab import *
from scipy import sparse

class QP(Optimizer):
    """
        This implementation does the following:

        minimize (1/2) x^T P x + q^T x
        subject to Gx <= h
                   Ax = b

        I heavily made use of this: https://osqp.org/docs/examples/mpc.html
    """
    def __init__(self):
        super().__init__()

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

        # Quadratic Objective:
        P = sparse.block_diag([sparse.kron(np.eye(dynamics.N), dynamics.Q),\
                        dynamics.Q, sparse.kron(np.eye(dynamics.N), \
                                dynamics.R)], format = 'csc')
        print(P.shape)
        # Linear Objective:
        q = np.hstack([np.kron(np.ones(dynamics.N), -dynamics.Q.dot(np.array(dynamics.ym))), \
                                -dynamics.Q.dot(dynamics.ym), \
                                    np.zeros(dynamics.N*nu)])
        q = q[:P.shape[0]]

        # Linear Dynamics:
        Ax = sparse.kron(sparse.eye(dynamics.N+1), - sparse.eye(nx)) + sparse.kron(sparse.eye(dynamics.N+1, k=-1), Ad)
        Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, dynamics.N)), sparse.eye(dynamics.N)]), Bd)

        Aeq = sparse.hstack([Ax, Bu])
        leq = np.hstack([-state, np.zeros(dynamics.N*nx)])

        Aineq = sparse.eye((dynamics.N+1)*nx + dynamics.N*nu)
        lineq = np.hstack([np.kron(np.ones(dynamics.N + 1), dynamics.xmin), np.kron(np.ones(dynamics.N), dynamics.umin)])
        uineq = np.hstack([np.kron(np.ones(dynamics.N+1), dynamics.xmax), np.kron(np.ones(dynamics.N), dynamics.umax)])

        # OSQP constraints:
        A = sparse.vstack([Aeq, Aineq], format='csc')
        l = np.hstack([leq, lineq])
        u = np.hstack([leq, uineq])

        prob = osqp.OSQP()
        prob.setup(P, q, A, l, warm_start=True)
        res = prob.solve()

        if res.info.status != 'solved':
            raise ValueError('OSQP solver did not solve the problem')

        u_optimal = res.x[-dynamics.N*nu:-(dynamics.N-1)*nu]

        return u_optimal




