#!/usr/bin/env python3
from dyno_model.car_driver_2D import *
from optimizer.newton_raphson import *
import matplotlib.pyplot as plt

# ym is target x and y
D2D = Driver2D(N1 = 0, N2= 5, Nu = 2, ym = [10.0, 10.0], K = 0.5, yn = [1.]*2, lambd = [1., 1.0], alpha = 30.0) # alpha is the speedup coefficient
D2D_opt = NewtonRaphson(cost= D2D.Cost, d_model= D2D)

new_state_new = [0, 0, 0, 0, 0, 0, 0]
start= [new_state_new[0], new_state_new[1]]

del_u = [0.0]*2
u = [0.05, 10.0]

sim_step = 0.0025

XY = []
Targ = []
state = []

for n in range(int(1000)):

    # D2D.ym  reference model won't change
    new_state_old = new_state_new

    D2D.state = np.array(new_state_new).flatten()

    state += [D2D.state[[0, 2]]]

    D2D.compute_cost(u, del_u)

    future_outputs = D2D.future_outputs(u, del_u)

    D2D.yn = future_outputs

    u_optimal = np.reshape(D2D_opt.optimize(u, del_u, True)[0], (-1, 1))

    del_u =  np.array(u_optimal.flatten()) - u

    del_u = del_u.flatten().tolist()

    u_optimal = np.array(u_optimal).flatten().tolist()

    new_state_new = D2D.predict(u = u_optimal, del_u = del_u, T = sim_step)

    u = u_optimal

    D2D.state = new_state_new

    XY += [D2D.yn]

    Targ += [D2D.ym]

XY = np.reshape(XY, (-1, 2))
Targ = np.reshape(Targ, (-1, 2))
state = np.reshape(state, (-1, 2))

labels = ["X", "Y"]

verts = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]])
plt.plot(state[:, 0], state[:, 1], '--k')
plt.scatter(Targ[:, 0], Targ[:, 1], c= 'r', marker = (5,2))
plt.scatter(start[0], start[1], c='b', marker = (5, 2))
plt.title("Position in XY Coordinates")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.show()












