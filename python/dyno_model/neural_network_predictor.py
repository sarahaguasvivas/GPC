from  .dynamic_model import *
from functions.functions import *
from cost.neural_network_cost import NN_Cost
from constraints.constraints import *
from keras import layers
from keras.models import load_model
import keras
import numpy as np
from keras import backend as K
import tensorflow as tf

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

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

        self.output_size = self.model.layers[-1].output_shape[1]
        self.input_size = self.model.layers[0].input_shape[1]
        self.hd = len(self.model.layers) - 1
        self.nd = 3
        self.dd = 1
        self.Hessian = np.zeros((self.output_size, self.output_size))

        """
        These attributes will be part of the recursion:
        """
        self.previous_first_der = 1
        self.previoud_second_der = 1

        super().__init__()
        self.Cost = NN_Cost(self, self.lambd)

    def __Phi_prime(self, x = 0):
        """
        Linear output function
        """
        return 1.0

    def __Phi_prime_prime(self, x = 0):
        """
        Linear output function
        """
        return 0.0

    """
    ---------------------------------------------------------------------

    Soloway, D., and P.J. Haley, “Neural Generalized Predictive Control,”
    Proceedings of the 1996 IEEE International Symposium on Intelligent
    Control, 1996, pp. 277–281.

    Calculating h'th element of the Jacobian
    Calculating m'th and h'th element of the Hessian

    ---------------------------------------------------------------------
    """

    def __partial_2_fnet_partial_nph_partial_npm(self, h, m, j):
        """
         D2^2f_j(net)
         ------------
        Du(n+h)Du(n+m)
        """
        return self.__Phi_prime()*self.__partial_2_net_partial_u_nph_partial_npm(h, m,  j)*\
                         + self.__Phi_prime_prime() * self.__partial_net_partial_u(h, j) * \
                                  self.__partial_net_partial_u(m, j)



    def __partial_2_yn_partial_nph_partial_npm(self, h, m, j):
        """
             D^2yn
        ---------------
        Du(n+h) Du(n+m)

        """
        weights = self.model.layers[j].get_weights()[0]
        hid = weights.shape[1]
        sum_output=0.0
        for i in range(hid):
            sum_output+= weights[j,i] * self.__partial_2_fnet_partial_nph_partial_npm(h, m, j)

        self.previous_second_der = sum_output
        return sum_output

    def __partial_2_net_partial_u_nph_partial_npm(self, h, m, j):
        """
          D^2 net_j
        -------------
        Du(n+h)Du(n+m)
        """
        weights = self.model.layers[j].get_weights()[0]
        sum_output=0.0
        for i in range(1, min(self.K, self.dd)):
            sum_output+= weights[j, i+self.nd+1] * self.previous_second_der * step(self.K-i-1)
        return sum_output

    def __partial_yn_partial_u(self, h, j):
        """
           D yn
        -----------
         D u(n+h)
        """
        weights = self.model.layers[j].get_weights()[0]
        hid = self.model.layers[j].output_shape[1]
        sum_output = 0.0
        for i in range(hid):
            sum_output += weights[j, i] * self.__partial_fnet_partial_u( h, j)

        self.previous_first_der = sum_output
        return sum_output

    def __partial_fnet_partial_u(self, h, j):
        """
        D f_j(net)
        ---------
         D u(u+h)
        """
        return self.__Phi_prime()*self.__partial_net_partial_u(h, j)

    def __partial_net_partial_u(self, h, j):
        """
         D net_j
        ---------
        D u(n+h)
        """
        weights = self.model.layers[j].get_weights()[0]
        self.nd = weights.shape[1] - 1
        sum_output = 0.0
        for i in range(self.nd):
            if (self.K - self.Nu) < i:
                sum_output+= weights[j, i+1] * kronecker_delta(self.K - i, h)
            else:
                sum_output+=weights[j, i+1] * kronecker_delta(self.Nu, h)

        for i in range(1, min(self.K, self.dd)):
            sum_output+= weights[j, i+self.nd+1] * self.previous_first_der * \
                                                 step(self.K - i -1)
        return sum_output


    def __partial_delta_u_partial_u(self, j, h):
        """
        D delta u
        ---------
        D u(n+h)

        """
        return kronecker_delta(h, j) - kronecker_delta(h, j-1)

    def compute_hessian(self, u, del_u):
        Hessian = np.zeros((self.Nu, self.Nu))
        for h in range(self.Nu):
            for m in range(self.Nu):
                sum_output=0.0

                for j in range(self.N1, self.N2):
                    sum_output += 2.*(self.__partial_yn_partial_u(h, j)*self.__partial_yn_partial_u(m, j) - \
                                        self.__partial_2_yn_partial_nph_partial_npm(h, m, j)* \
                                        (self.ym[j] - self.yn[j]))

                for j in range(self.Nu):
                    sum_output += 2.*( self.lambd[j] * (self.__partial_delta_u_partial_u(j, h) * self.__partial_delta_u_partial_u(j, m) + del_u[j] * 0.0))


                for j in range(self.Nu):
                    sum_output += kronecker_delta(h, j) * kronecker_delta(m, j) * \
                                        ( 2.0*self.constraints.s/( u[j] + self.constraints.r/2. - \
                                                self.constraints.b)**3 + 2.*self.constraints.s/(self.constraints.r/2. + \
                                                                    self.constraints.b - u[j])**3)

                Hessian[m, h] = sum_output

        return Hessian

    def compute_jacobian(self, u, del_u):
        # working on this now
        dJ = []
        for h in range(self.Nu):
            sum_output=0.0
            for j in range(self.N1, self.N2):
                sum_output+=-2.*(self.ym[j]-self.yn[j])*self.__partial_yn_partial_u(h, j)

            for j in range(self.Nu):
                sum_output+=2.*self.lambd[j]*del_u[j]*self.__partial_delta_u_partial_u(j, h)

            for j in range(self.Nu):
                sum_output+=kronecker_delta(h, j) * ( -self.constraints.s/(u[j] + self.constraints.r/2. - self.constraints.b)**2  + \
                                            self.constraints.s / (self.constraints.r/2. + self.constraints.b - u[j])**2    )

            dJ+=[sum_output]
        return dJ

    def Fu(self, u, del_u):
        jacobian = self.compute_jacobian(u, del_u)
        return jacobian

    def Ju(self, u, del_u):
        self.Hessian = self.compute_hessian(u, del_u)
        return self.Hessian

    def compute_cost(self, del_u, u):
        return self.Cost.compute_cost(del_u, u)

    def measure(self, u):
        if (u.ndim == 1):
            u = np.array([u])
        model_signal = load_model('../model_data/neural_network_1.hdf5')
        measure = model_signal.predict(u, batch_size=1)
        return measure

    def predict(self, x):
        return self.model.predict(x, batch_size=1)

