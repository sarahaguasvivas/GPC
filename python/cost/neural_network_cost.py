from .cost import *
from dyno_model.neural_network_predictor import *
import constraints.constraints as constraints

class NN_Cost(Cost):
    def __init__(self, dynamic_model, lambd : list):
        """
        N1      : minimum costing horizon
        N2      : maximum costing horizon
        Nu      : control horizon
        ym      : reference trajectory
        lambd   : control input weighting factor (damper for u(n+1))
        s       : sharpness corners of constraint function
        r       : range of constraint
        b       : offset of the range
        """
        self.d_model = dynamic_model
        self.N1 = dynamic_model.N1
        self.N2 = dynamic_model.N2
        self.Nu = dynamic_model.Nu
        self.ym = dynamic_model.ym
        self.yn = dynamic_model.yn
        self.lambd = dynamic_model.lambd
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
        print("ym: ", self.ym, "yn: ", self.yn)
        self.cost = 0.0

        self.ym = self.d_model.ym
        self.yn = self.d_model.yn

        # FIXME : this is supposed to be from N1 to N2
        self.cost+= (self.ym[0] - self.yn[0])
        angle_diff = (self.ym[1] - self.yn[1])
        if angle_diff > np.pi:
            angle_diff -= 2*np.pi
        if angle_diff < -np.pi:
            angle_diff += 2*np.pi
        self.cost += angle_diff

        for j in range(self.Nu):
            self.cost += (self.ym[j] - self.yn[j])**2

        for j in range(self.Nu):
            self.cost += self.lambd[j]*(del_u[j])**2

        for j in range(self.Nu):
            self.cost += self.s / (u[j] + self.r / 2.0 - self.b) + self.s / (self.r/2.0 + self.b - u[j]) - 4.0 / self.r

        return self.cost







