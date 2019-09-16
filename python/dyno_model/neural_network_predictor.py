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
        self.nd = 9
        self.dd = 1
        self.Hessian = np.zeros((self.input_size, self.input_size))
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

    def __jacobian_tensorflow(self, x):
        jacobian_matrix = []
        for m in range(self.output_size):
            grad_func = tf.gradients(self.model.output[:, m], self.model.input)
            gradients = sess.run(grad_func, feed_dict={self.model.input: x.reshape((1, x.size))})
            jacobian_matrix.append(gradients[0][0,:])

        return np.array(jacobian_matrix)

    def __hessian_tensorflow(self, x):
        hessian_matrix = []
        gradients=[]
        for m in range(self.output_size):
            temp=[]
            for n in range(self.output_size):
                grad_func = tf.gradients(self.model.output[:, m], self.model.input)
                gradients = sess.run(grad_func, feed_dict={self.model.input: x.reshape((1, x.size))})
                gradients = gradients[0][0, :]
                grad_func_1 = tf.gradients(self.model.output[:, n], self.model.input)
                gradients = sess.run(grad_func_1, feed_dict={self.model.input: gradients.reshape((1, gradients.size))})
                temp += [gradients[0][0, :]]
            temp = [tf.constant(0, dtype=tf.float32) if t is None else t for t in temp]
            temp = tf.stack(temp)
            hessian_matrix.append(temp)

        hessian_matrix = tf.stack(hessian_matrix)
        hessian_matrix = np.sum(hessian_matrix.eval(), axis = 2)
        return hessian_matrix

    def __partial_2_fnet_partial_nph_partial_npm(self, n, h, j):
        pass


    def __partial_2_yn_partial_nph_partial_npm(self, n, h, j):
        hid = self.model.layers[j].output_shape[1]
        weights = self.model.layers[j].get_weights()[0]
        sum_output = 0.0
        for i in range(hid):
            sum_output+=weights[j][i] # TODO

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
        weights = self.model.layers[-1].get_weights()[0]
        biases = self.model.layers[-1].get_weights()[1]
        sum_output= [0.0]*self.Nu
        for h in range(self.Nu):
            for j in range(self.model.layers[-1].input_shape[1]):
                for i in range(weights.shape[1]):
                    if (self.K - self.Nu) < i:
                        sum_output[h] += weights[j, i] * kronecker_delta(self.K-i, h)
                    else:
                        sum_output[h] += weights[j, i] * kronecker_delta(self.Nu, h)
        return sum_output

    def Fu(self, u):
        u = [0.0] + u.tolist()
        u = np.array(u)
        signal = self.measure(u)
        u = np.reshape(u[1:], (1, -1))
        u = np.concatenate((signal, u), axis = 1)
        jaco= np.sum(self.__jacobian_tensorflow(u), axis = 1)
        return jaco

    def Ju(self, u):
        u = [0.0] + u.tolist()
        u = np.array(u)
        signal = self.measure(u)

        u = np.reshape(u[1:], (1, -1))
        u = np.concatenate((signal, u), axis = 1)
        #grad =[]

        #for m in range(self.output_size):
        #    grad_func = tf.gradients(self.model.output[:, m], self.model.input)
        #    gradients = sess.run(grad_func, feed_dict={self.model.input: u.reshape((1, u.size))})
        #    grad += [gradients[0][0, :]]

        hess = self.__hessian_tensorflow(u)

        #grad = np.array(grad)
        #hessians = self.__hessian_tensorflow(u)
        #print(hessians.shape)
        #hess = self.__jacobian_tensorflow(grad)
        #print(hess.shape)
        #if len(hess.shape) > 2:
        #    ma_axis = np.argmax(hess.shape)
        #    hess = np.sum(hess, axis = ma_axis)

        #hess = np.transpose(hess)
        hess = np.array(hess)
        print(hess.shape)
        self.Hessian = hess
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

