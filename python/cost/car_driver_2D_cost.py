from .cost import *
from dyno_model.car_driver_2D import *
import constraints.constraints as constraints

class Driver2DCost(Cost):
    def __init__(self, dynamic_model):
        """
        N1      : minimum costing horizon
        N2      : maximum costing horizon
        Nu      : control horizon
        ym      : reference trajectory
        s       : sharpness corners of constraint function
        r       : range of constraint
        b       : offset of the range
        """
        self.d_model = dynamic_model
        self.s = self.d_model.constraints.s
        self.r = self.d_model.constraints.r
        self.b = self.d_model.constraints.b
        self.cost= 0.0
        super().__init__()

    def compute_cost(self, del_u : list, u : list):
        """
        del_u is a list of the element wise differences between current and
        previous control inputs
        n is an int that represents the current discrete timestep
        """
        self.cost = 0.0

        # FIXME : this is supposed to be from N1 to N2
        for j in range(self.Nu):
            self.cost += (self.ym[j] - self.yn[j])

        for j in range(self.Nu):
            self.cost += self.s/(u[j] + self.r / 2.0 - self.b) + self.s / (self.r/2.0 + self.b - u[j]) - 4.0 / self.r

        return self.cost







