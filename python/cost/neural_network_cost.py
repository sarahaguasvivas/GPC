from .cost import *
from dyno_model.neural_network_predictor import *
import constraints.constraints as constraints

class NN_Cost(Cost):
    def __init__(self, dynamic_model : NeuralNetworkPredictor, lambd : list, constraint : constraints):
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
        self.N2 = dynamic_model.N2
        self.Nu = dynamic_model.Nu
        self.ym = dynamic_model.ym
        self.yn = dynamic_model.yn
        self.lambd = lambd
        self.s = constraint.s
        self.r = constraint.r
        self.b = constraint.b
        self.cost= 0.0
        super().__init__()

    def compute_cost(self, n : int, del_u : list, u : list):
        """
        del_u is a list of the element wise differences between current and
        previous control inputs
        n is an int that represents the current discrete timestep
        """
        self.cost = 0.0
        for j in range(self.N1, self.N2):
            self.cost += (self.ym[n + j] - self.yn[n+j])**2

        for j in range(self.Nu):
            self.cost += self.lambd[j][del_u[n+j]]**2

        for j in range(self.Nu):
            self.cost += s/(u[n+j] + r / 2.0 - b) + s / (r/2.0 + b - u[n+j]) - 4.0 / r

        return self.cost







