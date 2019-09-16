#!/usr/bin/env python3
from dyno_model.neural_network_predictor import *
from optimizer.newton_raphson import *
import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

filename= "../model_data/neural_network_2.hdf5"

NNP = NeuralNetworkPredictor(model_file = filename, N1 = 5, N2= 10, Nu = 3, ym = [0.0]*3, K = 5, yn = [1.]*3, lambd = [0.1]*5)
NR_opt = NewtonRaphson(cost= NNP.Cost, d_model= NNP)

new_state_new = np.random.multivariate_normal([0.05]*12, 1.5*np.eye(12), 1)
du = [.5]*3

for n in range(1000):

    future_outputs = NNP.predict(new_state_new)

    future_outputs = future_outputs.flatten()

    new_state_old = new_state_new

    u_optimal = np.reshape(NR_opt.optimize(n, du, future_outputs, False)[0], (-1, 1))
    print(u_optimal)
    du = np.array(new_state_old[:, -3:]) - np.array(u_optimal.flatten())

    du = du.flatten()

    u_optimal = np.array([u_optimal])

    u_optimal = np.concatenate(([[[0.0]]], u_optimal), axis=1)

    signals = NNP.measure(u_optimal[:, :, 0])

    new_state_new = np.concatenate((signals, np.reshape(u_optimal.flatten()[1:], (1, -1))), axis=1)












