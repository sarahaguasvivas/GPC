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

        self.cornering_stiffness = 35000 #magnus paper

        self.Fcr = 0.
        self.Fcf= 0.

        self.front_wheel_spacing: float = 1.218 #.8 * self.length / 2  # distance of front wheels from center of mass
        self.rear_wheel_spacing: float = 1.628 #.8 * self.length / 2  # distance of rear wheels from center of mass

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
            x_6 : partial_xi_partial_u (1x2 vector)

            inputs:
            u[0] : acceleration
            u[1] : steering

        """
        C = np.zeros((2, 6)) # partial h(eps)/partial eps * partial eps / partial u

        self.__update_cornering_forces(self.state, u)

        C[0, 0] = 1
        C[1, 1] = 1

        return np.dot(C, np.dot(self._partial_f_partial_xi(self.state, u) , x_6) + self._partial_f_partial_u(self.state, u))

    def Ju(self, u, del_u):
        return self.compute_jacobian(u, del_u)

    def Fu(self, u, del_u):
        future_state = self.predict(u, del_u, self.K).tolist()
        print("future_state: ", future_state)
        return [future_state[0], future_state[1]]

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

        steps = np.arange(0.0, T, T / 10.0) # We're predicting after K steps

        delta_ode_state = odeint(self.__integrator, old_state, steps, args= (u, del_u))

        self.state = np.array(delta_ode_state[-1])

        return self.state

    def compute_cost(self, u, del_u):
        # abstract
        return self.Cost.compute_cost(u, del_u)

    def __update_cornering_forces(self, state, u):
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5, _ = state
        self.Fcf = self.cornering_stiffness * (steering - np.arctan((x_3 + self.front_wheel_spacing*x_5) / x_2))
        self.Fcr = - self.cornering_stiffness * np.arctan((x_3- self.rear_wheel_spacing*x_5) / x_2)

    def _partial_f_partial_xi(self, state, u):
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5, _  = state

        return np.array([[0, 0, np.cos(x_4), -np.sin(x_4), -x_2*np.sin(x_4) - x_3*np.cos(x_4), 0],
                [0, 0, np.sin(x_4), np.cos(x_4), x_2*np.cos(x_4) - x_3*np.sin(x_4), 0],
                [0, 0, 0, x_5, 0, x_3],
                [0, 0, -x_5, 0, 0, -x_2],
                [0, 0, 0, 0, 0, 1]])

    def _partial_f_partial_u(self, state, u):
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5, _ = state

        return np.array([[0, 0],
                        [0, 0],
                        [0, 1],
                        [-2./self.mass*self.Fcf*np.sin(steering), 0],
                        [0, 0],
                        [-2./self.yaw_inertia*self.front_wheel_spacing*self.Fcf*np.sin(steering), 0]])

    def __integrator(self, state, t,  u, del_u):
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

        x_0, x_1, x_2, x_3, x_4, x_5, x_6 = state

        self.__update_cornering_forces(state, u)

        #x_2 = max(x_2, 0.0) # ensuring TODO: see if I need it
        x_0_dot = x_2*np.cos(x_4) - x_3*np.sin(x_4)
        x_1_dot = x_2*np.sin(x_4) + x_3*np.cos(x_4)
        x_2_dot = x_5*x_3 + acceleration
        x_3_dot = -x_5*x_2 + 2./self.mass*(self.Fcf * np.cos(steering_angle) + self.Fcr)
        x_4_dot =  x_5
        x_5_dot = 2./self.yaw_inertia * (self.front_wheel_spacing * self.Fcf * np.cos(steering_angle) - self.rear_wheel_spacing*self.Fcr)
        x_6_dot = np.dot(self._partial_f_partial_xi(self.state, u) ,  x_6) + self._partial_f_partial_u(self.state, u)
        return [x_0_dot, x_1_dot, x_2_dot, x_3_dot, x_4_dot, x_6_dot]

    def future_outputs(self, u, del_u):
        # abstract
        # x is a vector with the sensor measurements and the current moves:
        future = self.Fu(u, del_u)
        return future

