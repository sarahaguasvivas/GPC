#!/usr/bin/env python3
from dyno_model.neural_network_predictor import *
from optimizer.newton_raphson import *
import sys
import os
import matplotlib.pyplot as plt
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

filename= "../model_data/neural_network_2.hdf5"

NNP = NeuralNetworkPredictor(model_file = filename, N1 = 0, N2= 2, Nu = 3, ym = [2., 0.0], K = 5, yn = [0.]*2, lambd = [0.1]*3)
NR_opt = NewtonRaphson(cost= NNP.Cost, d_model= NNP)

new_state_new = np.random.multivariate_normal([0.0]*12, 1.5*np.eye(12), 1)
new_state_new[:, -1] = 0.0
new_state_new[:, -2] = 1.0
new_state_new[:, -3] = 0.0

du = [0.0]*3

elapsed=[]
u_optimal_list=[]
ym = []
state = []

for n in range(50):
    seconds = time.time()

    future_outputs = NNP.predict(new_state_new).flatten()

    NNP.yn[0] = future_outputs[0]
    NNP.yn[1] = np.arctan2(future_outputs[1], future_outputs[2])

    new_state_old = new_state_new

    u_optimal = np.reshape(NR_opt.optimize(n, du, future_outputs, False)[0], (-1, 1))

    u_optimal_list+=[u_optimal.flatten().tolist()]

    du = np.array(new_state_old[:, -3:].flatten()) - np.array(u_optimal.flatten())

    du = du.flatten()

    u_optimal = np.array([u_optimal])

    u_optimal = np.concatenate(([[[0.0]]], u_optimal), axis=1)

    signals = NNP.measure(u_optimal[:, :, 0])

    new_state_new = np.concatenate((signals, np.reshape(u_optimal.flatten()[1:], (1, -1))), axis=1)

    elapsed += [time.time() - seconds]

    print("yn: ", NNP.yn, " time elapsed: ", elapsed[-1], " [s]")
    ym += [NNP.ym]
    state += [new_state_new[:, -3], np.arctan2(new_state_new[:, -2], new_state_new[:, -1])]

elapsed = np.array(elapsed).flatten()
ym = np.reshape(ym, (-1, 2))
state = np.reshape(state, (-1, 2))
u_optimal_list = np.reshape(u_optimal_list, (-1, 3))

plt.subplot(3, 1, 1)
plt.plot(elapsed)

plt.subplot(3, 1, 2)
plt.plot(ym[:, 0], '--k', label= 'target')
plt.plot(state[:, 0], 'r', label = 'state')

plt.legend()
plt.ylabel("block distance")

plt.subplot(3, 1, 3)
plt.plot(ym[:, 1], '--k', label='target')
plt.plot(state[:, 1], 'r', label = 'state')
plt.ylabel("twist sine")

plt.show()

