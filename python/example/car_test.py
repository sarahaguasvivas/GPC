#!/usr/bin/env python3
from dyno_model.car_driver_2D import *
from optimizer.newton_raphson import *
from cost.car_driver_2D_cost import *

D2D = Driver2D(N1 = 5, N2= 10, Nu = 6, ym = [0.0]*2, K = 5, yn = [1.]*2, lambd = [0.3]*5)
D2D.cost = Driver2DCost(D2D, D2D.lambd)
D2D_opt = NewtonRaphson(cost= D2D.cost, d_model= D2D)

new_state_new = np.random.multivariate_normal([.5]*6, 1.5*np.eye(6), 1)

del_u = [0.0]*2

for n in range(100):

    D2D.state = new_state_new.flatten().tolist()

    future_outputs = D2D.predict(D2D.state, del_u)

    D2D.yn[0] = D2D.state[0]
    D2D.yn[1] = D2D.state[2]

    future_outputs = future_outputs.flatten()

    new_state_old = new_state_new

    u_optimal = np.reshape(D2D_opt.optimize(future_outputs, del_u, True)[0], (-1, 1))

    del_u = np.array(new_state_old) - np.array(u_optimal.flatten())

    del_u = del_u.flatten()

    u_optimal = np.array([u_optimal])

    signals = D2D.measure(u_optimal[:, :, 0])


