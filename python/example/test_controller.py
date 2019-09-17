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

NNP = NeuralNetworkPredictor(model_file = filename, N1 = 0, N2= 2, Nu = 3, ym = [5., 0.0, 1.0], K = 5, yn = [0.]*3, lambd = [0.1]*3)
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
    print(n)
    seconds = time.time()

    future_outputs = NNP.predict(new_state_new)

    NNP.yn = future_outputs.flatten()

    future_outputs = future_outputs.flatten()

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
    state += [new_state_new[:, -3:]]


elapsed = np.array(elapsed).flatten()
ym = np.reshape(ym, (-1, 3))
state = np.reshape(state, (-1, 3))
u_optimal_list = np.reshape(u_optimal_list, (-1, 3))

plt.subplot(4, 1, 1)
plt.plot(elapsed)

plt.subplot(4, 1, 2)
plt.plot(ym[:, 0], '--k', label= 'target')
plt.plot(state[:, 0], 'k', label = 'state')
plt.plot(u_optimal_list[:, 0], 'r', label = 'control input')
plt.legend()
plt.ylabel("block distance")

plt.subplot(4, 1, 3)
plt.plot(ym[:, 1], '--k', label='target')
plt.plot(state[:, 1], 'k', label = 'state')
plt.plot(u_optimal_list[:, 1], 'r', label = 'control_input')
plt.ylabel("twist sine")

plt.subplot(4, 1, 4)
plt.plot(ym[:, 2], '--k', label = 'target')
plt.plot(state[:, 2], 'k', label = 'state')
plt.plot(u_optimal_list[:, 2], 'r', label = "control_input")
plt.ylabel('twist cosine')
plt.legend()
plt.xlabel('timestamp')
plt.show()






