#!/usr/bin/env python3
from dyno_model.car_ltv_mpc import *
from optimizer.qp import *
from cost.car_mpc import *
import matplotlib.pyplot as plt

############### TUNING PARAMS: ##########################
xmin = None
xmax = None

" u = [accel, steering]"
umin = [-10., -np.pi/4.0]
umax = [10., np.pi/4.0]

MAX_SIM_STEPS = 100
TARGET_THRESHOLD = 5.

Q = np.array([[150., 0], [0, 10.]]) # 3x3 in paper need to check
R = 9e-1*np.eye(2)
N = 10
Nc = 5
mu = 0.3
T = 0.05
rho = 1e3
Radius = 15. # m
#########################################################

# Using parameters in Predictive Active Steering Control for Autonomous Vehicle Systems;
# https://borrelli.me.berkeley.edu/pdfpub/pub-2.pdf

D2D = Driver2DMPC(N = N, Nc = Nc, dt = T, mu = mu, rho = rho, \
                            umin = umin, umax= umax, xmin = xmin,\
                                    xmax = xmax, Q=Q, R=R)
Cost = Driver2DCost(D2D)
QP = QP()

u_optimal = np.array([10.0, 0.1])

state_new_ode = [0.0, -Radius, 0.0, 0.0, 0.0, 0.0]

start= [state_new_ode[0], state_new_ode[1]]
state_new_linear = state_new_ode

D2D.state = state_new_ode

start_state = state_new_linear
del_u = np.zeros((2, 6))

target = []
state = []
ctrl=[]

way_point = 0

for i in range(MAX_SIM_STEPS):

    print("Cost : ", D2D.Cost.cost)
    print("state : ", D2D.state[:2])
    print("target : ", D2D.ym)

    if (D2D.Cost.cost < TARGET_THRESHOLD) and i > 0:
        way_point += 10.
        starting_state = D2D.state[:2]

    D2D.ym = [Radius * np.cos((way_point + 5)/100 - np.pi/2.0), \
                Radius * np.sin((way_point + 5)/100 - np.pi/2.0)]

    # Making Move:
    D2D.state = D2D.predict_ode(u_optimal, del_u, D2D.dt)

    D2D.compute_cost()

    state_new_ode = D2D.state

    del_u = D2D.get_optimal_control(QP, D2D.state, [0.0, 0.0])

    u_optimal = np.array(u_optimal) + 100*np.array(del_u)

    u_optimal = u_optimal[0]

    state  += [D2D.state]
    target += [D2D.ym]
    ctrl += [u_optimal]

state = np.reshape(state, (-1, 6))
target = np.reshape(target, (-1, 2))
ctrl = np.reshape(ctrl, (-1, 2))

plt.figure()
plt.subplot(1, 3, 1)
plt.plot(target[:, 0], target[:, 1], 'or', label = 'target')
plt.plot(state[:, 0], state[:, 1], 'k', label = 'trajectory', linewidth = 2, alpha = 0.2)
plt.plot(start[0], start[1], 'ob', label ='start')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(ctrl[:, 0], 'b', label = 'delta acceleration')
plt.legend()

plt.subplot(1,3,3)
plt.plot(ctrl[:, 1], 'b', label = 'delta steering')
plt.legend()
plt.show()

