from  .dynamic_model import *
from cost.car_driver_2D_cost import *
from functions.functions import *
from constraints.constraints import *
import numpy as np
import math
from scipy.integrate import odeint
import copy

class Driver2D(DynamicModel):
    def __init__(self, ym : list, K : int, \
                        yn : list, alpha : float):
        self.alpha = alpha

        self.ym = ym

        self.state = None
        self.yn = yn
        self.K = K

        self.constraints = Constraints(s = 1., r = 1., b = 1.)

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
        C[0, 0] = 1
        C[1, 1] = 1

        self.__update_cornering_forces(self.state, u)

        return np.dot(C, self.par_eps_par_u)

    def Ju(self, u, del_u):
        return self.compute_jacobian(u, del_u)

    def Fu(self, u, del_u):
        future_state, _ = self.predict(u, del_u, self.K)
        future_state = future_state.tolist()
        fu = [0.0, 0.0]
        fu[0] = self.ym[0] - future_state[0]
        temp = self.ym[1] - future_state[1]
        fu[1] = (temp + np.pi) % (2.*np.pi) - np.pi
        return np.array(fu)

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
        old_state = copy.copy(self.state)
        steps = np.arange(0.0, T, T / 10.0) # We're predicting after T steps
        delta_ode_state = odeint(self.__integrator, old_state, steps, args= (u,del_u))
        delta_ode_parepsparu = odeint(self.__integrator_2, self.par_eps_par_u.flatten(), steps, args=(u, del_u, delta_ode_state[-1]))
        return delta_ode_state[-1], np.reshape(delta_ode_parepsparu[-1], (-1, 2))

    def compute_cost(self, u, del_u):
        return self.Cost.compute_cost(u, del_u)

    def __update_cornering_forces(self, state, u):
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5 = state
        eps = 1e-8

        self.Fcf = self.corn_stiff * (steering - np.arctan((x_3 + self.lf*x_5) / (x_2 + eps)))
        self.Fcr = - self.corn_stiff * np.arctan((x_3 - self.lr*x_5) / (x_2 + eps))

    def _partial_f_partial_xi(self, state, u):
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5 = state
        eps = 1e-8
        """
            [ 0, 0, cos(x4),  -sin(x4), - x3*cos(x4) - x2*sin(x4),0]
            [ 0, 0, sin(x4),  cos(x4),   x2*cos(x4) - x3*sin(x4),0]
            [ 0, 0, 0, x5, 0, x3]
            [ 0, 0, (2*((C*(x3 - lr*x5))/(x2^2*((x3 - lr*x5)^2/x2^2 + 1)) + (C*cos(u1)*(x3 + lf*x5))/(x2^2*((x3 + lf*x5)^2/x2^2 + 1))))/m - x5, -(2*(C/(x2*((x3 - lr*x5)^2/x2^2 + 1)) + (C*cos(u1))/(x2*((x3 + lf*x5)^2/x2^2 + 1))))/m, 0,   (2*((C*lr)/(x2*((x3 - lr*x5)^2/x2^2 + 1)) - (C*lf*cos(u1))/(x2*((x3 + lf*x5)^2/x2^2 + 1))))/m - x2]
            [ 0, 0, 0,  0,   0,    1]
            [ 0, 0, -(2*((C*lr*(x3 - lr*x5))/(x2^2*((x3 - lr*x5)^2/x2^2 + 1)) - (C*lf*cos(u1)*(x3 + lr*x5))/(x2^2*((x3 + lr*x5)^2/x2^2 + 1))))/Iz, (2*((C*lr)/(x2*((x3 - lr*x5)^2/x2^2 + 1)) - (C*lf*cos(u1))/(x2*((x3 + lr*x5)^2/x2^2 + 1))))/Iz, 0, -(2*((C*lr^2)/(x2*((x3 - lr*x5)^2/x2^2 + 1)) + (C*lf*lr*cos(u1))/(x2*((x3 + lr*x5)^2/x2^2 + 1))))/Iz]

        """
        if x_2 == 0:
            x_2+= eps
        if x_5 == 0:
            x_5 +=eps
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
                                (self.corn_stiff*self.lf*np.cos(steering)*(x_3 + self.lr*x_5))/((x_2+eps)**2*((x_3 + self.lr*x_5)**2/(x_2**2+eps) + 1))))/self.yaw_inertia, \
                                        (2.*((self.corn_stiff*self.lr)/((x_2+eps)*((x_3 - self.lr*x_5)**2/(x_2**2+eps) + 1)) - \
                                        (self.corn_stiff*self.lf*np.cos(steering))/(eps + x_2*((x_3 + self.lr*x_5)**2/(x_2+eps)**2 + \
                                        1))))/self.yaw_inertia, 0, -(2.*((self.corn_stiff*self.lr**2)/(x_2*((x_3 - \
                                        self.lr*x_5)**2/(x_2+eps)**2 + 1)) + (self.corn_stiff*self.lf*self.lr*np.cos(steering))/(x_2*((x_3 +
                                        self.lr*x_5)**2/(x_2+eps)**2 + 1))))/self.yaw_inertia]])

    def _partial_f_partial_u(self, state, u):
        """
            [ 0,                                                                 0]
            [ 0,                                                                 0]
            [ 1,                                                                 0]
            [ 0,        (2*(C*cos(u1) - C*sin(u1)*(u1 - atan((x3 + lf*x5)/x2))))/m]
            [ 0,                                                                 0]
            [ 0, (2*(C*lf*cos(u1) - C*lf*sin(u1)*(u1 - atan((x3 + lr*x5)/x2))))/Iz]
        """
        acceleration, steering = u
        x_0, x_1, x_2, x_3, x_4, x_5 = state
        eps= 1e-8
        return np.array([[0, 0],
                        [0, 0],
                        [1, 0],
                        [0, (2.*(self.corn_stiff*np.cos(steering) - \
                                self.corn_stiff*np.sin(steering)*(steering -\
                                np.arctan((x_3 + self.lf*x_5)/(x_2+eps)))))/self.mass],
                        [0, 0],
                        [0, (2.*(self.corn_stiff*self.lf*np.cos(steering) - \
                                self.corn_stiff*self.lf*np.sin(steering)*(steering - \
                                np.arctan((x_3 + self.lr*x_5)/(x_2+eps)))))/self.yaw_inertia]])

    def __integrator_2(self, par_eps_par_u, t, u, del_u, state):
        """
        This is the integrator to calculate equation (24)

        """
        self.__update_cornering_forces(state, u)
        par_eps_par_u = np.reshape(par_eps_par_u, (-1, 2))
        x_6_dot = np.dot(self._partial_f_partial_xi(state, u) ,  par_eps_par_u) + self._partial_f_partial_u(state, u)

        return x_6_dot.flatten()

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

        x_0, x_1, x_2, x_3, x_4, x_5 = state

        self.__update_cornering_forces(state, u)
        x_2 = max(0, x_2)
        x_0_dot = x_2*np.cos(x_4) - x_3*np.sin(x_4)
        x_1_dot = x_2*np.sin(x_4) + x_3*np.cos(x_4)
        x_2_dot = x_5*x_3 + acceleration
        x_3_dot = -x_5*x_2 + 2./self.mass * (self.Fcf * np.cos(steering_angle) + self.Fcr)
        x_4_dot =  x_5
        x_5_dot = 2./self.yaw_inertia * (self.lf * self.Fcf * np.cos(steering_angle) - self.lr*self.Fcr)

        dFdt =  [x_0_dot, x_1_dot, x_2_dot, x_3_dot, x_4_dot, x_5_dot]

        return dFdt

    def future_outputs(self, u, del_u):
        future_state, _ = self.predict(u, del_u, self.K)
        return future_state.tolist()[:2]

