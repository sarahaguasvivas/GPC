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

    def compute_cost(self, u : list, del_u : list):
        """
        del_u is a list of the element wise differences between current and
        previous control inputs
        n is an int that represents the current discrete timestep
        """
        self.ym = self.d_model.ym
        self.yn = self.d_model.yn

        #for j in range(2):
        #    self.cost += (self.ym[j] - self.yn[j])**2

        #for j in range(2):
        #    self.cost += del_u[j]**2

        #for j in range(2):
        #    self.cost += self.s / (u[j] + self.r / 2.0 - self.b) + self.s / (self.r/2.0 + self.b - u[j]) - 4.0 / self.r

        self.cost= np.linalg.norm(np.array(self.d_model.ym) - np.array(self.d_model.yn))
        print("Cost: ",  self.cost)
        return self.cost







