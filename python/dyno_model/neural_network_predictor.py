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
        self.nd = 2
        self.d_d = 3
        self.constraints = Constraints()
        self.model = load_model(model_file)

        super().__init__()
        self.Cost = NN_Cost(self, self.lambd)

    def _first_term(self, h, m, u, n):
        # Removed n from u because u will rewrite itself
        # to have the specific window
        first_term = 0.0
        for j in range(self.Nu):
            first_term += kronecker_delta(h, j)* kronecker_delta(m, j) * ( 2.0*self.Cost.s/(u[j] + \
                                            self.Cost.r/2.0 - self.Cost.b)**3  + 2.0*self.Cost.s/ (self.Cost.r /2.0 + \
                                                                    self.Cost.b - u[j])**3  )
        return first_term

    def _second_term(self, h, m, n):
        second_term = 0.0
        for j in range(self.Nu):
            second_term += self.Cost.lambd[j] * (kronecker_delta(h, j) - kronecker_delta(h, j-1)) * \
                                            (kronecker_delta(m, j) - kronecker_delta(m, j-1))
        return 2.*second_term

    def __output_function_derivative(self):
        """
        Linear output function
        """
        return 1.0

    def __output_function_second_derivative(self):
        """
        Linear output function
        """
        return 0.0

    def __partial_yn_partial_u(self, n, h, m):
        # because my output function is linear
        return 0.0

    def _third_term(self, h, m, n):
        return 0.0

    def compute_hessian(self, n,  del_u, u):
        for i in range(self.Nu):
            for j in range(self.Nu):
                self.Hessian[i, j] = self._first_term(i, j, u, n) + self._second_term(i, j, n) + \
                                                    self._third_term(i, j, n)
        return self.Hessian


    def compute_function(self, del_u, u, n):
        sum_s= 0.0
        for j in range(self.N2):
            sum_s+= (self.ym[n+j] - self.yn[n+j])**2

        for j in range(self.Nu):
            sum_s += self.Cost.lambd[j] * del_u[n+j]**2 + \
                     self.Cost.s / (u[n+j] + self.Cost.r/2.0 - self.Cost.b) + \
                                self.Cost.s / (self.Cost.r / 2.0 +  \
                                        self.Cost.b - u[n+j]) - 4.0/self.Cost.r
        return sum_s

    def measure(self, u):
        model_signal = load_model('../model_data/neural_network_1.hdf5')
        measure = model_signal.predict(u, batch_size=1)
        return measure

    def predict(self, x):
        # x is a vector with the sensor measurements and the current moves:
        return model.predict(x, batch_size=1)

