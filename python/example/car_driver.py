#!/usr/bin/env python3
from dyno_model.car_driver_2D import *
from optimizer.newton_raphson import *
import matplotlib.pyplot as plt

############### TUNING PARAMS: ##########################
MAX_ACCEL= 10.
MIN_ACCEL = -5.0
MAX_STEERING = np.pi/4.0 - 0.2
R = 15.
SIM_STEP = 0.0025
MAX_SIM_STEPS = 10000
MAX_NR_IT = 8 # Maximum Newton Raphson Iterations
ALPHA= 30.
TARGET_THRESHOLD = 0.2
#########################################################

D2D = Driver2D(ym = [0.0, 0.0], K = .2, yn = [0.0, 0.0], alpha = ALPHA)
D2D_opt = NewtonRaphson(cost= D2D.Cost, d_model= D2D)

del_u = np.array([0.0, 0.0])
u_optimal = np.array([0.0, 0.0])

state_new = np.random.multivariate_normal(mean = [0.05]*6, cov = 0.05*np.eye(6), size = 1).flatten().tolist()

state_new[0] = R * np.cos(-np.pi/2.0)
state_new[1] = R * np.sin(-np.pi/2.0)

D2D.state = state_new

start= state_new[:2]
XY = [D2D.state[:2]]
targ = []

starting_state = start
way_point = 0

for n in range(MAX_SIM_STEPS):
    D2D.compute_cost(u_optimal, del_u)

    if (D2D.Cost.cost < TARGET_THRESHOLD) and n > 0:
        way_point += 1
        starting_state = D2D.state[:2]

    D2D.ym[0] =  R * np.cos((way_point + 5)/1000 - np.pi/2.0)
    D2D.ym[1] =  R * np.sin((way_point + 5)/1000 - np.pi/2.0)

    u_optimal_old = u_optimal

    # Applying input to controller
    D2D.state, D2D.par_eps_par_u = D2D.predict(u = u_optimal, del_u = del_u, T = SIM_STEP)

    Fu = D2D.Fu(u_optimal, del_u)
    Ju = D2D.Ju(u_optimal, del_u)

    try:
        # Solving Integral through Euler Method
        u_optimal =  u_optimal + SIM_STEP * D2D.alpha * np.dot(np.linalg.inv(Ju), Fu) # should it be + u_optimal?
    except:
        u_optimal = np.array([0.0, 0.0])

    u_optimal[0] = np.clip(u_optimal[0], MIN_ACCEL, MAX_ACCEL)
    u_optimal[1] = np.clip(u_optimal[1], -MAX_STEERING, MAX_STEERING)

    print("u_optimal : ", u_optimal.flatten().tolist())

    del_u = np.array(u_optimal.flatten()) - np.array(u_optimal_old)

    u_optimal = np.array(u_optimal).flatten().tolist()

    D2D.yn = D2D.state[:2]
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











