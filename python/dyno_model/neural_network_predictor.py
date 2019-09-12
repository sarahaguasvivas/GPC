from  .dynamic_model import *
from functions.functions import *
from cost.neural_network_cost import NN_Cost
from constraints.constraints import *
from keras import layers
from keras.models import load_model
import keras
import numpy as np
from keras import backend as K

class NeuralNetworkPredictor(DynamicModel):
    def __init__(self, model_file : str, N1 : int, \
                                                N2 : int, Nu : int, ym : list, K : int, \
                                                    yn : list, lambd : list):
        self.N1 = N1
        self.N2 = N2
        self.Nu = Nu
        self.ym = ym
        self.lambd = lambd
        self.Hessian = np.zeros((self.Nu, self.Nu))
        self.yn = yn
        self.K = K
        self.num_predicted_states = 3
        self.constraints = Constraints()
        self.model = load_model(model_file)

        print(self.model.summary())
        print(self.model.get_config())

        self.hd = len(self.model.layers) - 1
        self.nd = 9
        self.dd = 1

        super().__init__()
        self.Cost = NN_Cost(self, self.lambd)

    def __Phi_prime(self, x = 0):
        """
        Linear output function
        """
        return [1.0]*(self.hd + 1)

    def __Phi_prime_prime(self, x = 0):
        """
        Linear output function
        """
        return [0.0]*(self.hd + 1)


    def __parial_yn_partial_u(self, n, h, j):
        hid = self.model.layers[j].output_shape[1]
        weights = self.model.layers[j].get_weights()[0]
        sum_output = 0.0
        for i in range(hid):
            sum_output += weights[j][i] * self.__partial_net_partial_u(n, h, i)

        return sum_output


    def __partial_fnet_partial_u(self, n, h, j):
        phi_prime = self.__Phi_prime()
        phi_prime_prime = self.__Phi_prime_prime()

        return phi_prime*self.__partial_net_partial_u(n, h, j)

    def __partial_net_partial_u(self, n, h, j):
        """
        n   ->     timestep number
        h   ->     dummy index
        j   ->     hidden layer
        """
        weights = self.model.layers[j].get_weights()[0]
        sum_first = 0.0
        for i in range(self.nd):
            if (self.K - self.Nu)<i:
                sum_first += weights[j][i+1] * kroenecker_delta(self.K - i, h)
            else:
                sum_first += weights[j][i+1] * kroenecker_delta(self.Nu, h)

        return sum_first

    def __partial_yn_partial_u(self, n, h):
        pass


    def compute_hessian(self, n,  del_u, u):
        pass


    def compute_jacobian(self, n, del_u, u):
        # abstract
        weights = self.model.layers[-1].get_weights()[0]
        biases = self.model.layers[-1].get_weights()[1]
        sum_output= [0.0]*self.Nu
        for h in range(self.Nu):
            for j in range(self.model.layers[-1].input_shape[1]):
                for i in range(self.nd):
                    if (self.K - self.Nu) < i:
                        sum_output[h] += weights[j, i] * kronecker_delta(self.K-i, h)
                    else:
                        sum_output[h] += weights[j, i] * kronecker_delta(self.Nu, h)
        return sum_output

    def compute_cost(self, del_u, u):
        # abstract
        return self.Cost.compute_cost(del_u, u)

    def measure(self, u):
        # abstract
        model_signal = load_model('../model_data/neural_network_1.hdf5')
        measure = model_signal.predict(u, batch_size=1)
        return measure

    def predict(self, x):
        # abstract
        # x is a vector with the sensor measurements and the current moves:
        return self.model.predict(x, batch_size=1)

