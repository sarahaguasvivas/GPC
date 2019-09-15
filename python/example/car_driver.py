#!/usr/bin/env python3
from dyno_model.car_driver_2D import *
from optimizer.newton_raphson import *

D2D = Driver2D(N1 = 5, N2= 10, Nu = 6, ym = [0.0]*2, K = 5, yn = [1.]*2, lambd = [0.3]*5)
D2D_opt = NewtonRaphson(cost= D2D.Cost, d_model= D2D)

new_state_new = np.random.multivariate_normal([.5]*6, 30*np.eye(6), 1)

del_u = [0.5]*2
u = [1.]*2

for n in range(1000):

    D2D.state = new_state_new.flatten().tolist()
    print(D2D.state)
    future_outputs = D2D.predict(n, u, del_u)

    future_outputs = future_outputs.flatten()

    new_state_old = new_state_new

    u_optimal = np.reshape(D2D_opt.optimize(n, del_u, future_outputs, False)[0], (-1, 1))

    del_u = np.array(new_state_old[:, -3:]) - np.array(u_optimal.flatten())

    del_u = del_u.flatten()

    u_optimal = np.array([u_optimal])

    u_optimal = np.concatenate(([[[0.0]]], u_optimal), axis=1)

    signals = D2D.measure(u_optimal[:, :, 0])

    new_state_new = np.concatenate((signals, np.reshape(u_optimal.flatten()[1:], (1, -1))), axis=1).flatten()











