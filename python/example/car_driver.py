#!/usr/bin/env python3
from dyno_model.car_driver_2D import *
from optimizer.newton_raphson import *
import matplotlib.pyplot as plt

D2D = Driver2D(N1 = 5, N2= 10, Nu = 2, ym = [1.0, 10.0], K = 0.1, yn = [1.]*2, alpha = 1e-8)
D2D_opt = NewtonRaphson(cost= D2D.Cost, d_model= D2D)

new_state_new = np.random.multivariate_normal([0, 0, 0, 0, 0, 0], 1.5*np.eye(6), 1).tolist()

del_u = [0.0]*2
u = [0.02, 10.0]

sim_step = 0.1

XY = []
Targ = []
state = []

for n in range(1000):

    # D2D.ym  reference model won't change
    new_state_old = new_state_new

    D2D.state = np.array(new_state_new).flatten()

    D2D.compute_cost(u, del_u)

    future_outputs = D2D.future_outputs(u, del_u)

    D2D.yn = future_outputs

    u_optimal = np.reshape(D2D_opt.optimize(u, del_u, True)[0], (-1, 1))

    del_u = u - np.array(u_optimal.flatten())

    del_u = del_u.flatten().tolist()

    u_optimal = np.array(u_optimal).flatten().tolist()

    new_state_new = D2D.predict(u = u_optimal, del_u = del_u, T = sim_step)

    u = u_optimal

    D2D.state = new_state_new

#    print("yn: ", D2D.yn, " ym: ", D2D.ym)

    XY += [D2D.yn]
    state += [D2D.state[[0, 2]]]
    Targ += [D2D.ym]

XY = np.reshape(XY, (-1, 2))
Targ = np.reshape(Targ, (-1, 2))
state = np.reshape(state, (-1, 2))

for i in range(XY.shape[1]):
    plt.subplot(1, 2, i+1)
    plt.plot(state[:, i], 'r')
    plt.plot(Targ[:, i], '--k')

plt.show()












