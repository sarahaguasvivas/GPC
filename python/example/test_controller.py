#!/usr/bin/env python3.5
from dyno_model.neural_network_predictor import *
from optimizer.newton_raphson import *

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

filename= "../model_data/neural_network_2.hdf5"

NNP = NeuralNetworkPredictor(model_file = filename, N1 = 5, N2= 10, Nu = 5, ym = [0.0]*10, K = 5, yn = [1.]*10, lambd = [0.3]*5)
NR_opt = NewtonRaphson(cost= NNP.Cost, d_model= NNP)

new_state_new = np.random.multivariate_normal([0.]*12, 1.5*np.eye(12), 5)
du = [0.]*5

for n in range(1000):

    future_outputs= NNP.predict(new_state_new)

    future_outputs = future_outputs.flatten()

    new_state_old = new_state_new

    u_optimal = NR_opt.optimize(n, du, future_outputs)

    du = (np.array(new_state_old[-2:]) - np.array(u_optimal)).tolist

    signals = NNP.measure(u_optimal).tolist

    new_state_new = signals + u_optimal.tolist












