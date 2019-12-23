from  .dynamic_model import *
from cost.car_mpc import *
from functions.functions import *
from constraints.constraints import *
from optimizer.qp import *

import numpy as np
import math
from scipy.integrate import odeint
import copy

class Driver2DMPC(DynamicModel):
    def __init__(self, N : int, Nc : int, \
            dt: float, mu : float, rho : float, \
            umin = None, umax=  None, xmin = None, xmax = None,
                        max_speed = None, \
                            Q = None, R = None):
        self.state = None
        self.ym = None
        self.yn = None

        self.dt = dt
        self.N = N    # prediction horizon
        self.Nc= Nc   # control horizon
        self.mu = mu
        self.rho = rho

        if max_speed is None:
            max_speed = np.inf
        else:
            max_speed = max_speed

        # Setting up constraints:
        if umin is None:
            self.umin = [-5, -np.pi/4.0]
            self.Nu = 2
        else:
            self.umin = umin
            self.Nu = len(umin)

        if umax is None:
            self.umax = [10, np.pi/4.0]
        else:
            self.umax = umax

        self.xmin = xmin
        self.xmax = xmax

        if Q is None:
            self.Q = np.eye(2)
        else:
            self.Q = Q

        if R is None:
            self.R = 150.
        else:
            self.R = R

        self.collided: bool = False
        self.length: float = 4.9  # a typical car length (Toyota Camry)
        self.width: float = 1.83  # a typical car width (Toyota Camry)
        self.mass: float = 1587  # Toyota Camry gross mass (kg)
        self.corn_stiff = 35000 #magnus paper
        self.par_eps_par_u =  np.zeros((6, 2)) # partial epsilon partial u
        self.Fcr = 0.
        self.Fcf= 0.
        self.lf: float = 1.218 #.8 * self.length / 2  # distance of front wheels from center of mass
        self.lr: float = 1.628 #.8 * self.length / 2  # distance of rear wheels from center of mass
        self.yaw_inertia: float = 2315.3
        self.road_friction_coefficient: float = .9  # .9 (dry road), .6 (wet road), .2 (snow), .05 (ice)
        self.slip_coefficient = 1.0
        self.max_speed = 200  # TODO: make configurable
        super().__init__()

        self.Cost = Driver2DCost(self)

    def Ju(self, u, del_u):
        pass

    def Fu(self, u, del_u):
        pass

    def _get_FGHM(self, state, u):
        """
            A = J_A    B = J_B    H = [x, y]    M = 0
            x(k+1) = F*x(k) + G*u(k)
            y(k+1) = H*x(k) + M*u(k)
        """
        A = self._partial_f_partial_xi(state, u)
        B = self._partial_f_partial_u(state, u)

        Hk = np.zeros((2, 6))
        Hk[0, 0] = 1.
        Hk[1, 1] = 1.

        # M = 0 so not counting
        Mk = np.zeros((6, 2))

        # Getting F_k:
        Acum = np.identity(A.shape[0])
        Sum = np.zeros(A.shape[0])
        for i in range(A.shape[0]):
            Sum= np.add(Sum, (Acum*self.dt**i)/np.math.factorial(i))
            Acum = np.dot(Acum, A)
        Fk = Sum

        # Getting G_k:
        Acum = np.identity(A.shape[0])
        Sum = np.zeros(A.shape[0])
        for i in range(1, A.shape[0]):
            Sum = np.add(Sum, (Acum*self.dt**(i)/np.math.factorial(i)))
            Acum = np.dot(Acum, A)
        Gk = np.dot(Sum, B)

        return Fk, Gk, Hk, Mk

    def predict_trajectory(self, u, del_u):
        """
            u --> current control input
            Del_U--> we are solving for it
            N are prediction steps (continuous)
            Nc are control steps (continuous)
        """
        state= self.state
        u = u
        predicted_trajectory = []
        for i in range(self.N):
            if i < self.Nc:
                F, G, H, M = self._get_FGHM(state, u + del_u[i])
                state = np.add(np.dot(F, state), np.dot(G, u + del_u[i])).tolist()
                u = u + del_u[i]
                predicted_trajectory+=[state]
            else:
                # del_u = 0 for k = t + Nc, ..., t + N - 1
                F, G, H, M = self._get_FGHM(state, u)
                state = np.add(np.dot(F, state), np.dot(G, u)).tolist()
                predicted_trajectory += [state]
        return predicted_trajectory

    def predict_measurements(self, predicted_trajectory):
        H = np.zeros((2, 6))
        H[0,0] = 1.
        H[1,1] = 1.
        predicted_trajectory = np.reshape(predicted_trajectory, (-1, 6))
        return np.dot(H, predicted_trajectory.T).T

    def get_optimal_control(self, QP, state, u):
        dynamics = self
        Del_U = QP.optimize(dynamics = dynamics, state = state, u0 = u)
        print("u_optimal: ", Del_U)
        return Del_U

    def compute_cost(self, u, del_u):
        return self.Cost.compute_cost(u, del_u)

    def __update_cornering_forces(self, state, u):
        """
            This updates the cornering forces using the
            given control inputs and the state of the model
        """
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5 = state
        eps = 1e-8
        self.Fcf = self.corn_stiff * (steering - np.arctan((x_3 + self.lf*x_5) / (x_2 + eps)))
        self.Fcr = - self.corn_stiff * np.arctan((x_3 - self.lr*x_5) / (x_2 + eps))

    def _partial_f_partial_xi(self, state, u):
        """
            partial f
            ---------- = A
            partial xi
        """
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5 = state
        eps = 1e-8
        if x_2 == 0:
            x_2+= eps
        if x_5 == 0:
            x_5 +=eps
        if x_3 == 0:
            x_3 +=eps
        return np.array([[0, 0, np.cos(x_4), -np.sin(x_4), -x_3*np.cos(x_4) - x_2*np.sin(x_4), 0],
                [0, 0, np.sin(x_4), np.cos(x_4), x_2*np.cos(x_4) - x_3*np.sin(x_4), 0],
                [0, 0, 0, x_5, 0, x_3],
                [0, 0, -x_5 + 2./self.mass*(np.cos(steering)*self.corn_stiff*((x_3+ \
                                    self.lr * x_5)/(eps+(x_3+self.lf*x_5)**2 + x_2**2)) + \
                                    self.corn_stiff*((x_3-self.lr*x_5)/(eps+(x_3-self.lr*x_5)**2 + x_2**2))), \
                                     -(2.*(self.corn_stiff/((x_2+eps)*((x_3 - self.lr*x_5)**2/(x_2**2+eps) + 1.)) + \
                                    (self.corn_stiff*np.cos(steering))/(eps + x_2*((x_3 + \
                                    self.lf*x_5)**2/(x_2**2+eps) + 1))))/self.mass, \
                                    0, \
                                    -(2.*(self.corn_stiff/((x_2+eps)*((x_3 - self.lr*x_5)**2/(x_2+eps)**2 + 1)) +\
                                    (self.corn_stiff*np.cos(steering))/((x_2+eps)*((x_3 + self.lf*x_5)**2/(x_2**2+eps) + 1))))/self.mass],
                [0, 0, 0, 0, 0, 1],
                [ 0, 0, -(2.*((self.corn_stiff*self.lr*(x_3 - self.lr*x_5))/((x_2+eps)**2*((x_3 - self.lr*x_5)**2/(x_2**2+eps) + 1.)) -\
                                (self.corn_stiff*self.lf*np.cos(steering)*(x_3 + self.lr*x_5))/((x_2+eps)**2*((x_3 + \
                                        self.lr*x_5)**2/(x_2**2+eps) + 1))))/self.yaw_inertia, \
                                        (2.*((self.corn_stiff*self.lr)/((x_2+eps)*((x_3 - self.lr*x_5)**2/(x_2**2+eps) + 1)) - \
                                        (self.corn_stiff*self.lf*np.cos(steering))/(eps + x_2*((x_3 + self.lr*x_5)**2/(x_2+eps)**2 + \
                                        1))))/self.yaw_inertia, 0, -(2.*((self.corn_stiff*self.lr**2)/(eps + x_2*((x_3 - \
                                        self.lr*x_5)**2/(x_2+eps)**2 + 1)) + (self.corn_stiff*self.lf*self.lr*np.cos(steering))/(eps+x_2*((x_3 +
                                        self.lr*x_5)**2/(x_2+eps)**2 + 1))))/self.yaw_inertia]])

    def _partial_f_partial_u(self, state, u):
        """
            partial f
            ---------
            partial u
        """
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5 = state
        eps= 1e-8
        if x_2 == 0:
            x_2+= eps
        return np.array([[0, 0],
                        [0, 0],
                        [1, 0],
                        [0, (2.*(self.corn_stiff*np.cos(steering) - \
                                self.corn_stiff*np.sin(steering)*(steering -\
                                np.arctan((x_3 + self.lf*x_5)/(x_2)))))/self.mass],
                        [0, 0],
                        [0, (2.*(self.corn_stiff*self.lf*np.cos(steering) - \
                                self.corn_stiff*self.lf*np.sin(steering)*(steering - \
                                np.arctan((x_3 + self.lr*x_5)/(x_2)))))/self.yaw_inertia]])
    def predict(self, u):
        """
            IN  : dynamic model (self), u
            OUT : vector of predicted states

            Predicts a state starting from the current state
            within the prediction horizon (N)
        """
        predicted_state = self.state
        for i in range(self.N):
            F, G, H, M = self._get_FGHM(predicted_state, u)
            predicted_state = np.add(np.dot(F, self.state), np.dot(G, u))
        return predicted_state

    ############################      WILL DELETE SOON BELOW      ###################################

    def predict_ode(self, u, del_u, T):
        """
            states:
            x_0 : x, x_1 : y,  x_2 : x_dot, x_3 : y_dot, x_4 : yaw, x_5 : yaw_dot

            inputs:
            u[0] : acceleration, u[1] : steering
        """
        old_state = copy.copy(self.state)
        steps = np.arange(0.0, T, T / 10.) # We're predicting after T steps
        delta_ode_state = odeint(self.__integrator, old_state, steps, args= (u,del_u))
        self.__update_cornering_forces(delta_ode_state[-1], u)
        return np.array(delta_ode_state[-1]).flatten()

    def __integrator(self, state, t, u, del_u):
        """
            states:
                x_0 : x, x_1 : y, x_2 : x_dot, x_3 : y_dot, x_4 : yaw, x_5 : yaw_dot
            inputs:
                u[0] : acceleration, u[1] : steering
        """
        acceleration, steering_angle = u

        x_0, x_1, x_2, x_3, x_4, x_5 = state
        eps= 1e-8
        self.Fcf = self.corn_stiff * (steering_angle - np.arctan((x_3 + self.lf*x_5) / (x_2 + eps)))
        self.Fcr = - self.corn_stiff * np.arctan((x_3 - self.lr*x_5) / (x_2 + eps))
        x_2 = max(x_2, 0.0)
        x_0_dot = x_2*np.cos(x_4) - x_3*np.sin(x_4)
        x_1_dot = x_2*np.sin(x_4) + x_3*np.cos(x_4)
        x_2_dot = x_5*x_3 + acceleration
        x_3_dot = -x_5*x_2 + 2./self.mass * (self.Fcf * np.cos(steering_angle) + self.Fcr)
        x_4_dot =  x_5
        x_5_dot = 2./self.yaw_inertia * (self.lf * self.Fcf * np.cos(steering_angle) - self.lr*self.Fcr)
        dFdt =  [x_0_dot, x_1_dot, x_2_dot, x_3_dot, x_4_dot, x_5_dot]
        return dFdt


