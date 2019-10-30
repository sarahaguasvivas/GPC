#!/usr/bin/env python3
from dyno_model.car_driver_2D import *
from optimizer.newton_raphson import *
import matplotlib.pyplot as plt

# ym is target x and y

D2D = Driver2D(N1 = 0, N2= 5, Nu = 2, ym = [2.5, 2.5], K = 0.5, yn = [1.]*2, lambd = [1., 1.], alpha = 30.) # alpha is the speedup coefficient

D2D_opt = NewtonRaphson(cost= D2D.Cost, d_model= D2D)

new_state_new = np.random.multivariate_normal([0.0]*18, .5*np.eye(18), 1).flatten().tolist() # nonzero x_2 to avoid nan in first calculation of Fcr and Fcf

start= [new_state_new[0], new_state_new[1]]

del_u = [0.000]*2
u = [20., 0.00]

sim_step = 0.0025

R = 200.

XY = [new_state_new[:2]]
Targ = []

for n in range(100):

#    D2D.ym[0] = R * np.cos(n/10)
#    D2D.ym[1] = R * np.sin(n/10)

    new_state_old = new_state_new

    D2D.state = np.array(new_state_new).flatten()

    D2D.compute_cost(u, del_u)

    future_outputs = D2D.future_outputs(u, del_u)

    D2D.yn = future_outputs

    u_optimal = np.reshape(D2D_opt.optimize(u = u, del_u = del_u, rtol = 1e-8, maxit = 8, verbose= False)[0], (-1, 1))
    print("u_optimal :", u_optimal)
    del_u = u - np.array(u_optimal.flatten())

    del_u = del_u.flatten().tolist()

    u_optimal = np.array(u_optimal).flatten().tolist()

    new_state_new = D2D.predict(u = u_optimal, del_u = [0.0,0.0], T = sim_step)

    D2D.yn = new_state_new[:2]

    if (abs(D2D.Cost.cost) < 0.02):
        break

    u = u_optimal

    D2D.state = np.array(new_state_new).flatten()

    XY += [D2D.yn]
    Targ += [D2D.ym]

XY = np.reshape(XY, (-1, 2))
Targ = np.reshape(Targ, (-1, 2))

labels = ["X", "Y"]

#verts = np.array([[-1, -1], [1, -1], [1, 1], [-1, -1]])
plt.plot(XY[:, 0], XY[:, 1], '--k')
plt.scatter(Targ[:, 0], Targ[:, 1], c= 'r', marker = (5,2))
plt.scatter(start[0], start[1], c='b', marker = (5, 2))
plt.title("Position in XY Coordinates")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")

plt.show()












