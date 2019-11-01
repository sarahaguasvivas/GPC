#!/usr/bin/env python3
from dyno_model.car_driver_2D import *
from optimizer.newton_raphson import *
import matplotlib.pyplot as plt

############### TUNING PARAMS: ###################
MAX_ACCEL= 40.
MAX_STEERING = np.pi / 4.0
R = 10.
SIM_STEP = 0.1
MAX_NR_IT = 8 # Maximum Newton Raphson Iterations
##################################################

D2D = Driver2D(ym = [0.0, 0.0], K = 0.5, yn = [0.0, 0.0], alpha = 30.)

D2D_opt = NewtonRaphson(cost= D2D.Cost, d_model= D2D)

del_u = np.array([0.0, 0.0])

u_optimal = np.array([MAX_ACCEL, 0.0])

state_new = np.random.multivariate_normal(mean = [0.05]*18, cov = 0.05*np.eye(18), size = 1).flatten().tolist()

D2D.state = state_new
start= state_new[:2]
XY = [D2D.state[:2]]
targ = []

for n in range(10):

    D2D.ym[0] =  R                                      # R * np.cos(n/1000 - np.pi/2.0)
    D2D.ym[1] =  0.0                                    # R * np.sin(n/1000 - np.pi/2.0)

    state_old = state_new
    u_optimal_old = u_optimal

    D2D.state = np.array(state_new).flatten()

    u_optimal = np.reshape(D2D_opt.optimize(u = [0.0, 0.0], del_u = [0.0, 0.0], rtol = 1e-8, \
                            maxit = MAX_NR_IT, verbose= True)[0], (-1, 1))

    u_optimal[0] = np.clip(u_optimal[0], 0, 40)
    u_optimal[1] = np.clip(u_optimal[1], -np.pi/4.0, np.pi/4.0)

    print("u_optimal : ", u_optimal.flatten().tolist())

    del_u = u_optimal_old - np.array(u_optimal.flatten())

    u_optimal = np.array(u_optimal).flatten().tolist()

    state_new = D2D.predict(u = u_optimal, del_u = [0.0, 0.0], T = SIM_STEP)

    D2D.yn = state_new[:2]

    targ += [D2D.ym][0]
    XY += [D2D.yn]

######################### PLOTTING TRAJECTORIES: ##############################
XY = np.reshape(XY, (-1, 2))
targ = np.reshape(targ, (-1, 2))

labels = ["X", "Y"]
plt.plot(XY[:, 0], XY[:, 1], '--k', label = 'position')
plt.scatter(targ[:, 0], targ[:, 1], c = 'r', marker = (5,2), label = 'target')
plt.scatter(start[0], start[1], c = 'b', marker = (5, 2), label = 'start')
plt.title("Position in XY Coordinates")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
plt.legend()
plt.show()
##############################################################################











