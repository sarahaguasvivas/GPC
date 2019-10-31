#!/usr/bin/env python3
from dyno_model.car_driver_2D import *
from optimizer.newton_raphson import *
import matplotlib.pyplot as plt

MAX_ACCEL= 10.
MAX_STEERING = np.pi/4.0
R = 200.

D2D = Driver2D(N1 = 0, N2= 2, Nu = 2, ym = [1.5, 1.5], K = 0.5, yn = [R*np.cos(-np.pi/2.0), R*np.sin(-np.pi/2.0)], alpha = 30.) # alpha is the speedup coefficient

D2D_opt = NewtonRaphson(cost= D2D.Cost, d_model= D2D)

new_state_new = np.random.multivariate_normal([0.05]*18, 0.05*np.eye(18), 1).flatten().tolist() # nonzero x_2 to avoid nan in first calculation of Fcr and Fcf

del_u = [0.0001]*2
u = [10., 0.0]

sim_step = 0.025

new_state_new[0] = R * np.cos(-np.pi/2.0)
new_state_new[1] = R * np.sin(-np.pi/2.0)

D2D.yn = new_state_new[:2]

start= [new_state_new[0], new_state_new[1]]

XY = [new_state_new[:2]]
Targ = []

for n in range(10):

    D2D.ym[0] = R * np.cos(n/100 - np.pi/2.0)
    D2D.ym[1] = R * np.sin(n/100 - np.pi/2.0)

    Targ += [D2D.ym][0]

    new_state_old = new_state_new

    D2D.state = np.array(new_state_new).flatten()

    D2D.compute_cost(u, del_u)

    future_outputs = D2D.future_outputs(u, del_u)

    D2D.yn = future_outputs

    u_optimal = np.reshape(D2D_opt.optimize(u = u, del_u = del_u, rtol = 1e-8, maxit = 8, verbose= False)[0], (-1, 1))

    accl= u_optimal[0]

    if (accl>0):
        u_optimal[0] = min(MAX_ACCEL, u_optimal[0])
    if (accl<0):
        u_optimal[0] = max(0.0, u_optimal[0])

    if u_optimal[1] > 0:
        u_optimal[1] = min(MAX_STEERING, u_optimal[1]) - np.pi/2.0 # making the inputs more realistic
    if u_optimal[1] < 0:
        u_optimal[1] = max(-MAX_STEERING, u_optimal[1]) + np.pi/2.0

    print("u_optimal : ", u_optimal.flatten().tolist())

    del_u = u - np.array(u_optimal.flatten())

    del_u = del_u.flatten().tolist()

    u_optimal = np.array(u_optimal).flatten().tolist()

    new_state_new = D2D.predict(u = u_optimal, del_u = del_u, T = sim_step)

    D2D.yn = new_state_new[:2]

    u = u_optimal

    D2D.state = np.array(new_state_new).flatten()

    XY += [D2D.yn]

XY = np.reshape(XY, (-1, 2))
Targ = np.reshape(Targ, (-1, 2))

labels = ["X", "Y"]

#verts = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]])
plt.plot(XY[:, 0], XY[:, 1], '--k', label = 'position')
plt.scatter(Targ[:, 0], Targ[:, 1], c= 'r', marker = (5,2), label = 'target')
plt.scatter(start[0], start[1], c='b', marker = (5, 2), label = 'start')
plt.title("Position in XY Coordinates")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.show()












