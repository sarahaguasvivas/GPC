from  .dynamic_model import *
from cost.car_driver_2D_cost import *
from functions.functions import *
from constraints.constraints import *
import numpy as np
import math
from scipy.integrate import odeint

class Driver2D(DynamicModel):
    def __init__(self, N1 : int, \
                        N2 : int, Nu : int, ym : list, K : int, \
                        yn : list, lambd : list, alpha : float):
        self.N1 = N1
        self.N2 = N2
        self.alpha = alpha

        self.Nu = Nu
        self.ym = ym

        self.lambd = lambd # might not need it

        self.state = [0., 0., 0., 0., 0., np.array([0, 0])]
        self.yn = yn
        self.K = K

        self.constraints = Constraints(s = 1e-10, r = 1, b = 1)

        self.collided: bool = False
        self.length: float = 4.9  # a typical car length (Toyota Camry)
        self.width: float = 1.83  # a typical car width (Toyota Camry)
        self.mass: float = 1587  # Toyota Camry gross mass (kg)

        self.corn_stiff = 35000 #magnus paper

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

    def compute_jacobian(self, u, del_u):
        """
        This is actually not a jacobian it is just the term
        partial g /partial u in this paper:
        https://magnus.ece.gatech.edu/Papers/NRFlowECC19.pdf

        This is Equation (25)
            states:
            x_0 : x
            x_1 : y
            x_2 : x_dot
            x_3 : y_dot
            x_4 : yaw
            x_5 : yaw_dot
            x_6 : partial_xi_partial_u (5x2 vector)

            inputs:
            u[0] : acceleration
            u[1] : steering

        """
        C = np.zeros((2, 6)) # partial h(eps)/partial eps * partial eps / partial u

        self.__update_cornering_forces(self.state, u)

        C[0, 0] = 1
        C[1, 1] = 1

        x_6 = np.reshape(self.state[6:], (-1, 2))
        return np.dot(C, x_6)

    def Ju(self, u, del_u):
        return self.compute_jacobian(u, del_u)

    def Fu(self, u, del_u):
        future_state = self.predict(u, del_u, self.K).tolist()
        return future_state[:2]

    def __dampen(self, val, lim, coef):
        damped = val*coef
        if np.abs(damped) < lim:
            return 0.0
        else:
            return damped

    def predict(self, u, del_u, T):
        """
            states:
            x_0 : x
            x_1 : y
            x_2 : x_dot
            x_3 : y_dot
            x_4 : yaw
            x_5 : yaw_dot
            x_6 : partial_xi_partial_u

            inputs:
            u[0] : acceleration
            u[1] : steering
        """

        old_state = self.state
        steps = np.arange(0.0, T, T / 10.0) # We're predicting after T steps
        delta_ode_state = odeint(self.__integrator, old_state, steps, args= (u,del_u))
        state = delta_ode_state[-1]
        return state

    def compute_cost(self, u, del_u):
        return self.Cost.compute_cost(u, del_u)

    def __update_cornering_forces(self, state, u):
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5 = state[:6]
        eps = 1e-8
        self.Fcf = self.corn_stiff * (steering - np.arctan((x_3 + self.lf*x_5) / (x_2+eps )))
        self.Fcr = - self.corn_stiff * np.arctan((x_3- self.lr*x_5) / (x_2 + eps))

    def _partial_f_partial_xi(self, state, u):
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5 = state[:6]
        eps = 1e-8
        return np.array([[0, 0, np.cos(x_4), -np.sin(x_4), -x_2*np.sin(x_4) - x_3*np.cos(x_4), 0], # checked.
                [0, 0, np.sin(x_4), np.cos(x_4), x_2*np.cos(x_4) - x_3*np.sin(x_4), 0], # checked.
                [0, 0, 0, x_5, 0, x_3], # checked.
                [0, 0, -x_2+2./self.mass*(np.cos(steering)*self.corn_stiff*((x_3+self.lf*x_5)/(x_2**2 + eps + (x_3+self.lf*x_5)**2)) - self.corn_stiff*((x_3-self.lr*x_5)/( eps + x_2**2 + (x_3-self.lr*x_5)**2))), 2./self.mass*(np.cos(steering)*self.corn_stiff*((-x_2**2)/(eps + x_2**2 + (x_3 + self.lr*x_5)**2)) - self.corn_stiff*(x_2**2)/(eps + x_2**2 + (x_3 -
                    self.lr*x_5)**2)), 0, -x_2+ 2./self.mass*(np.cos(steering)*self.corn_stiff)*(-self.lf*(x_2**2)/(x_2**2 + eps + (x_3 + self.lf*x_5)**2)+ self.corn_stiff* self.lr* (x_2**2)/(eps + x_2**2 + (x_3- self.lr*x_5)**2)) ], # checked.
                [0, 0, 0, 0, 0, 1],
                [0, 0, 2./self.yaw_inertia*(self.lf*np.cos(steering)*self.corn_stiff*((x_3+self.lf)/(eps + x_2**2 + (x_3 + self.lr*x_5)**2) + self.lr*self.corn_stiff*(x_3-self.lr*x_5)/(eps + x_2**2 + (x_3-self.lr*x_5)**2))), 2./self.yaw_inertia*(self.lf*np.cos(steering)*self.corn_stiff*(-x_2**2)/(eps + x_2**2 + (x_3 + self.lf * x_5)**2) + self.lr*self.corn_stiff*(x_2**2)/(eps + x_2**2 + (x_3-self.lr*x_5)**2)), 0,
                    2./self.yaw_inertia*(self.lf*np.cos(steering)*self.corn_stiff*(-self.lf*(x_2**2)/(eps + x_2**2 +
                    (x_3 + self.lf*x_5)**2)) - self.lr**2*self.corn_stiff * (x_2**2)/(eps + x_2**2 + (x_3- self.lr*x_5)**2))]])

    def _partial_f_partial_u(self, state, u):
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5 = state[:6]
        eps= 1e-8
        return np.array([[0, 0],
                        [0, 0],
                        [1, 0],
                        [0, -2./self.mass*( -np.sin(steering) *self.corn_stiff*(steering) - np.arctan((x_3+self.lf*x_5)/(x_2 + eps)) + np.cos(steering)*self.corn_stiff)],
                        [0, 0],
                        [0, -2./self.yaw_inertia * ( -self.lf*np.sin(steering)*self.corn_stiff*(steering - np.arctan((x_3+self.lr*x_5)/(x_2 + eps)))  + self.lf*np.cos(steering) * self.corn_stiff ) ]])

    def __integrator(self, state, t, u, del_u):
        """
        states:
            x_0 : x
            x_1 : y
            x_2 : x_dot
            x_3 : y_dot
            x_4 : yaw
            x_5 : yaw_dot
            x_6 : partial_xi_partial_u

        inputs:
            u[0] : acceleration
            u[1] : steering
        """
        acceleration, steering_angle = u

        x_0, x_1, x_2, x_3, x_4, x_5 = state[:6]
        x_6 = state[6:]

        self.__update_cornering_forces(state, u)
        x_3 = max(0, x_3)
        x_0_dot = x_2*np.cos(x_4) - x_3*np.sin(x_4)
        x_1_dot = x_2*np.sin(x_4) + x_3*np.cos(x_4)
        x_2_dot = x_5*x_3 + acceleration
        x_3_dot = -x_5*x_2 + 2./self.mass * (self.Fcf * np.cos(steering_angle) + self.Fcr)
        x_4_dot =  x_5
        x_5_dot = 2./self.yaw_inertia * (self.lf * self.Fcf * np.cos(steering_angle) - self.lr*self.Fcr)

        x_6 = np.reshape(x_6, (-1, 2))
        x_6_dot = np.dot(self._partial_f_partial_xi(self.state, u) ,  x_6) + self._partial_f_partial_u(self.state, u)

        dFdt =  [x_0_dot, x_1_dot, x_2_dot, x_3_dot, x_4_dot, x_5_dot]

        dFdt += x_6_dot.flatten().tolist()

        return dFdt

    def future_outputs(self, u, del_u):
        future = self.Fu(u, del_u)
        return future

