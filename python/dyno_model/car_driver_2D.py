from  .dynamic_model import *
from cost.car_driver_2D_cost import *
from functions.functions import *
from constraints.constraints import *
import numpy as np

class Driver2D(DynamicModel):
    def __init__(self, N1 : int, \
                        N2 : int, Nu : int, ym : list, K : int, \
                         yn : list, lambd : list):
        self.N1 = N1
        self.N2 = N2
        self.Nu = Nu
        self.ym = ym
        self.lambd = lambd
        self.state = [0.5]*self.Nu
        self.yn = yn
        self.K = K
        self.num_predicted_states = 3
        self.constraints = Constraints()

        self.collided: bool = False
        self.length: float = 4.9  # a typical car length (Toyota Camry)
        self.width: float = 1.83  # a typical car width (Toyota Camry)
        self.mass: float = 1500  # Toyota Camry gross mass (kg)

        self.front_wheel_spacing: float = .8 * self.length / 2  # distance of front wheels from center of mass
        self.rear_wheel_spacing: float = .8 * self.length / 2  # distance of rear wheels from center of mass
        self.yaw_inertia: float = 14229.7 * .11308
        self.road_friction_coefficient: float = .9  # .9 (dry road), .6 (wet road), .2 (snow), .05 (ice)

        self.max_steering_angle: 60 / 360.0 * np.pi * 2  # TODO: make configurable
        self.max_speed = 200  # TODO: make configurable

        self.half_car_length: float = self.length / 2
        self.half_car_width: float = self.width / 2

        super().__init__()
        self.Cost = Driver2DCost(self, self.lambd)


    def compute_dynamics(self, n, del_u, u):
        """
        As we represent the dynamics as
        x_dot = f(x, u)
        """

        pass

    def compute_hessian(self, n,  del_u, u):
        " No need for Hessian in this case. "
        pass

    def compute_jacobian(self, n, del_u, u):
        self.Jacobian = np.zeros((self.Nu, self.Nu))
        self.Jacobian[0, 1] = 1
        self.Jacobian[1, 3] = self.state[5]
        self.Jacobian[1, 5] = self.state[3]
        self.Jacobian[2, 3] = 1
        self.Jacobian[3, 1] = -self.state[5]
        self.Jacobian[3, 5] = -self.state[1]
        self.Jacobian[4, -1] = 1
        return self.Jacobian

    def Ju(self, n , del_u, u):
        return self.compute_jacobian(n, del_u, u)

    def Fu(self, del_u, u):
        """
        x0 = x
        x1 = x_dot
        x2 = y
        x3 = y_dot
        x4 = yaw
        x5 = yaw_dot

        u = [steering, acceleration]
        """
        x, x_dot, y, y_dot, yaw, yaw_dot = self.state


        steering_angle = u[0]
        acceleration = u[1] # acceleration

        # slip angles
        beta = np.arctan((self.rear_wheel_spacing / (self.front_wheel_spacing + \
                                    self.rear_wheel_spacing)) * np.tan(steering_angle))
        speed = np.sqrt(x_dot ** 2 + y_dot ** 2)
        slip_angle = (speed / self.rear_wheel_spacing) * np.sin(beta)
        front_slip_angle = -slip_angle
        rear_slip_angle = 0.0  # TODO: this is probably not realistic

        # tire cornering stiffness
        front_tire_cornering_stiffness = self.road_friction_coefficient * self.mass * (
                self.rear_wheel_spacing / (self.front_wheel_spacing + self.rear_wheel_spacing))

        Fcf = -front_tire_cornering_stiffness * front_slip_angle
        Fcr = -front_tire_cornering_stiffness * rear_slip_angle

        x_dot_dot = yaw_dot * y_dot + acceleration
        y_dot_dot = -yaw_dot * x_dot + (2 / self.mass) * \
                        (Fcf * np.cos(steering_angle) + Fcr)


        x_dot = max(x_dot, 0.0)  # Prevent reversing (longitudinal velocity)

        self.Fu = np.array([x, yaw_dot*y_dot + x_dot_dot, y_dot, \
                        -yaw_dot*x_dot + 2./self.mass*(Fcf*np.cos(front_slip_angle) + Fcr), \
                        yaw_dot, \
                        2./self.yaw_inertia*(self.front_wheel_spacing*Fcf - self.rear_wheel_spacing*Fcr)])

        return self.Fu

    def compute_cost(self, del_u, u):
        # abstract
        return self.Cost.compute_cost(del_u, u)

    def measure(self, u):
        # abstract
        pass


    def __integrator(self, n, u, del_u):
        """
            Integrator
        """

        steering_angle = u[0]
        acceleration = u[1]

        x, x_dot, y, y_dot, yaw, yaw_dot = self.state

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
            x_dot_dot, y_dot = x_dot_dot * ratio, y_dot_dot * ratio

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

        return np.array([dx, x_dot_dot, dy, y_dot_dot, yaw_dot, yaw_dot_dot])


    def predict(self,n,  u, del_u):
        # abstract
        # x is a vector with the sensor measurements and the current moves:
        return self.__integrator(n, u, del_u )

