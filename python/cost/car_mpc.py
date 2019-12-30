from .cost import *
from dyno_model.car_driver_2D import *
import constraints.constraints as constraints

class Driver2DCost(Cost):
    def __init__(self, d_model):
        """
        N1      : minimum costing horizon
        N2      : maximum costing horizon
        Nu      : control horizon
        ym      : reference trajectory
        s       : sharpness corners of constraint function
        r       : range of constraint
        b       : offset of the range
        """
        self.d_model = d_model
        self.cost= 0.0
        super().__init__()

    def compute_cost(self):
        """
        del_u is a list of the element wise differences between current and
        previous control inputs
        n is an int that represents the current discrete timestep
        """
        eta_diff = np.array(self.d_model.ym) - np.array(self.d_model.measure())
        self.cost = np.linalg.norm(eta_diff)
        return self.cost







