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

        eta_diff = np.array(self.yn) - np.array(self.yn)

        Q = self.d_model.Q
        R = self.d_model.R

        Del_u = np.reshape(del_u, (-1, 2))

        self.cost= np.dot(np.dot(np.dot(eta_diff.T ,  Q.T) , Q ) , eta_diff) + \
                            np.dot(np.dot(np.dot(np.array(Del_u).T,R.T) , R), Del_u)

        print("Cost: ",  self.cost)
        return self.cost







