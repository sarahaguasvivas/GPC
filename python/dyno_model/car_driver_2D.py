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
        self.lambd = lambd
        self.state = [0.0]*6
        self.yn = yn
        self.K = K
        self.constraints = Constraints(s = 1e-10, r = 1, b = 1)
        self.collided: bool = False
        self.length: float = 4.9  # a typical car length (Toyota Camry)
        self.width: float = 1.83  # a typical car width (Toyota Camry)
        self.mass: float = 1500  # Toyota Camry gross mass (kg)

        self.Fcr = 0.
        self.Fcf= 0.5

        self.front_wheel_spacing: float = .8 * self.length / 2  # distance of front wheels from center of mass
        self.rear_wheel_spacing: float = .8 * self.length / 2  # distance of rear wheels from center of mass
        self.yaw_inertia: float = 14229.7 * .11308
        self.road_friction_coefficient: float = .9  # .9 (dry road), .6 (wet road), .2 (snow), .05 (ice)
        self.slip_coefficient = 1.0
        self.max_steering_angle: 60 / 360.0 * np.pi * 2  # TODO: make configurable
        self.max_speed = 200  # TODO: make configurable

        self.half_car_length: float = self.length / 2
        self.half_car_width: float = self.width / 2

        super().__init__()
        self.Cost = Driver2DCost(self)

    def compute_hessian(self, u, del_u):
        pass

    def compute_jacobian(self, u, del_u):
        """
        This is actually not a jacobian it is just the term
        partial g /partial u in this paper:
        https://magnus.ece.gatech.edu/Papers/NRFlowECC19.pdf
        """
        steering_angle = u[0]
        acceleration = u[1]

        x, x_dot, y, y_dot, yaw, yaw_dot = self.state

        a = np.zeros((2, 6)) # partial h(eps)/partial eps * partial eps / partial u
        b = np.zeros((6, 2)) # partial eps / partial u

        # slip angles
        beta = np.arctan((self.rear_wheel_spacing / (self.front_wheel_spacing + \
                                    self.rear_wheel_spacing)) * np.tan(steering_angle))

        speed = np.sqrt(x_dot ** 2 + y_dot ** 2)

        slip_angle = (speed / self.rear_wheel_spacing) * np.sin(beta)

        front_slip_angle = - slip_angle

        rear_slip_angle = 0.0  # TODO: this is probably not realistic

        front_tire_cornering_stiffness = self.road_friction_coefficient * self.mass * (
                self.rear_wheel_spacing / (self.front_wheel_spacing + self.rear_wheel_spacing))

        self.Fcf = -front_tire_cornering_stiffness * front_slip_angle
        self.Fcr = -front_tire_cornering_stiffness * rear_slip_angle

        a[0, 1] = 1
        a[1, 3] = 1

        b[1, 1] = 1
        b[3, 0] = -2. / self.mass * self.Fcf * np.sin(steering_angle)

        self.Jacobian = self.alpha*np.dot(a, b)

        return self.Jacobian

    def Ju(self, u, del_u):
        return self.compute_jacobian(u, del_u)

    def Fu(self, u, del_u):
        future_state = self.predict(u, del_u, self.K).tolist()
        return [future_state[0], future_state[2]]

    def __dampen(self, val, lim, coef):
        damped = val*coef
        if np.abs(damped) < lim:
            return 0.0
        else:
            return damped

    def predict(self, u, del_u, T):
        """
        predictor!

        x0 = x
        x1 = x_dot
        x2 = y
        x3 = y_dot
        x4 = yaw
        x5 = yaw_dot

        u = [steering, acceleration]
        """
        x, x_dot, y, y_dot, yaw, yaw_dot = self.state

        old_state = self.state

        steering_angle = u[0]

        acceleration = u[1] # acceleration

        ode_state  = self.state

        steps = np.arange(0.0, T, T / 10.0) # We're predicting after K steps

        delta_ode_state = odeint(self.__integrator, ode_state, steps, args= (u, steering_angle, acceleration, del_u))

        self.state = np.array(delta_ode_state[-1])

        v_x = (x - old_state[0]) / T

        v_y = (y - old_state[2]) / T

        speed = np.sqrt(v_x**2 + v_y**2)

        # y_dot:
        self.state[3] = self.__dampen(self.state[3], lim=0.1, coef = self.slip_coefficient)

        # yaw_rate:
        self.state[5] = self.__dampen(self.state[5], lim=0.0017, coef = 0.95)

        return self.state

    def compute_cost(self, u, del_u):
        # abstract
        return self.Cost.compute_cost(u, del_u)

    def measure(self, u):
        # abstract
        pass

    def __integrator(self, state, t,  u, steering_angle, acceleration, del_u):
        """
            Integrator

        """
        x, x_dot, y, y_dot, yaw, yaw_dot = state

        x_dot = max(x_dot, 0.0)

        beta = np.arctan(
            (self.rear_wheel_spacing / \
             (self.front_wheel_spacing + \
              self.rear_wheel_spacing)) * \
            np.tan(steering_angle)
        )

        speed = np.sqrt(x_dot ** 2 + y_dot ** 2)

        slip_angle = (speed / self.rear_wheel_spacing) * np.sin(beta)

        front_slip_angle = - slip_angle
        rear_slip_angle = 0.0  # TODO find out expression that depends on frict coeff

        #### TIRE CORNERING STIFFNESS:

        front_tire_cornering_stiffness = self.road_friction_coefficient * \
                                         self.mass * (self.rear_wheel_spacing / \
                                                      (self.front_wheel_spacing + \
                                                       self.rear_wheel_spacing))

        #### CORNERING FORCES:
        front_tire_cornering_force = - front_tire_cornering_stiffness * \
                                     front_slip_angle
        rear_tire_cornering_force = - front_tire_cornering_stiffness * \
                                    rear_slip_angle

        #### ACCELERATION:
        x_dot_dot = yaw_dot * y_dot + \
                                    acceleration

        y_dot_dot = -yaw_dot * x_dot + \
                               (2.0 / self.mass) * \
                               (front_tire_cornering_force * \
                                np.cos(steering_angle) + \
                                rear_tire_cornering_force)

        #### CLAMP ACCELERATION IF ABOVE MIN VEL:
        speed = np.sqrt(
            (x_dot_dot + x_dot) ** 2 + \
            (y_dot_dot + y_dot) ** 2
        )

        if speed > self.max_speed:
            a = x_dot_dot ** 2 + y_dot_dot ** 2

            b = 2.0 * (x_dot_dot * x_dot + \
                       y_dot_dot * y_dot
                       )
            c = x_dot ** 2 + y_dot ** 2 - \
                self.max_speed ** 2

            sqrt_term = b ** 2 - 4.0 * a * c

            eps = 0.0000001
            if sqrt_term < eps:
                ratio = 0.0
            else:
                ratios = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2.0 * a), \
                         (-b - np.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)
                ratio = max(ratios)

            x_dot_dot, y_dot_dot = x_dot_dot * ratio, y_dot_dot * ratio

        yaw_dot_dot = (2.0 / self.yaw_inertia) * \
                           (self.front_wheel_spacing * \
                            front_tire_cornering_force - \
                            self.rear_wheel_spacing * \
                            rear_tire_cornering_force)

        dx = x_dot * np.cos(yaw) - \
                y_dot * np.sin(yaw)
        dy = x_dot * np.sin(yaw) + \
                y_dot * np.sin(yaw)

        #### CLAMP VELOCITY:
        speed = np.sqrt(dx ** 2 + dy ** 2)
        if speed > self.max_speed:
            ratio = self.max_speed / speed
            dx, dy = dx * ratio, dy * ratio

        return [dx, x_dot_dot, dy, y_dot_dot, yaw_dot, yaw_dot_dot]

    def future_outputs(self, u, del_u):
        # abstract
        # x is a vector with the sensor measurements and the current moves:
        future = self.Fu(u, del_u)
        return future

