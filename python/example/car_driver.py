#!/usr/bin/env python3
from dyno_model.car_driver_2D import *
from optimizer.newton_raphson import *

D2D = Driver2D(N1 = 5, N2= 10, Nu = 6, ym = [0.0]*2, K = 5, yn = [1.]*2, alpha = 1)
D2D_opt = NewtonRaphson(cost= D2D.Cost, d_model= D2D)

new_state_new = np.random.multivariate_normal([0.0]*6, 1.5*np.eye(6), 1).tolist()

del_u = [0.5]*2
u = [1.]*2

sim_step = 0.1

for n in range(1000):
    # D2D.ym  reference model won't change
    D2D.state = np.array(new_state_new).flatten()

    future_outputs = D2D.future_outputs(u, del_u)

    new_state_old = new_state_new

    u_optimal = np.reshape(D2D_opt.optimize(future_outputs, del_u, False)[0], (-1, 1))

    del_u = u - np.array(u_optimal.flatten())

    del_u = del_u.flatten().tolist()

    u_optimal = np.array(u_optimal).flatten().tolist()

    u = u_optimal

    new_state_new = D2D.predict(u = u_optimal, del_u = del_u, T = sim_step)

    D2D.yn = [new_state_new[0], new_state_new[2]]

    print("yn: ", D2D.yn, " ym: ", D2D.ym)











