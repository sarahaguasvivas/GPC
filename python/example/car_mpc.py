#!/usr/bin/env python3
from dyno_model.car_ltv_mpc import *
from optimizer.newton_raphson import *
import matplotlib.pyplot as plt

############### TUNING PARAMS: ##########################
MAX_ACCEL= 15.
MIN_ACCEL = -5.0
MAX_STEERING = np.pi/4.0 - 0.2
MAX_SIM_STEPS = 1000
MAX_NR_IT = 8 # Maximum Newton Raphson Iterations
TARGET_THRESHOLD = 0.2
#########################################################

D2D = Driver2DMPC(ym = [0.0, 0.0], N = .5, Nc = 1,  yn = [0.0, 0.0], dt = 0.02)

u_optimal = np.array([10.0, 0.0])

state_new_ode = np.random.multivariate_normal(mean = [0.0]*6, cov = 3.*np.eye(6), size = 1).flatten().tolist()

state_new_ode[0] = 0.0
state_new_ode[1] = 0.0

state_new_linear = state_new_ode
D2D.state = state_new_ode

start_state = state_new_linear

ode = []
linear = []

for i in range(MAX_SIM_STEPS):
    u_optimal[0] = 0.0
    u_optimal[1] = np.pi/4.0 * np.sin(i / D2D.dt)

    D2D.state = state_new_ode
    state_new_linear = D2D.predict(u_optimal)

    state_new_ode = D2D.predict_ode(u_optimal, [0.0, 0.0], D2D.dt)

    print(state_new_ode, state_new_linear)

    ode+=[state_new_ode]
    linear+=[state_new_linear]

ode = np.reshape(ode, (-1, 6))
linear = np.reshape(linear, (-1, 6))
errors = ode- linear
labels = ['$x$', '$y$', '$\dot{x}$', '$\dot{y}$', '$\psi$', '$\dot{\psi}$']

plt.figure()
plt.title('Linearized States vs. ODE States at dt = ' + str(D2D.dt) + ' sec')

for i in range(6):
    plt.subplot(6, 2, 2*i + 1)
    plt.plot(D2D.dt*np.arange(MAX_SIM_STEPS), ode[:, i], label='ode')
    plt.plot(D2D.dt*np.arange(MAX_SIM_STEPS), linear[:, i], label='linear')
    plt.ylabel(labels[i])
    max_ = np.max(ode[:, i])
    min_ = np.min(ode[:, i])
    plt.ylim((min_ - 10, max_ + 10))
    plt.legend()

for i in range(6):
    plt.subplot(6, 2, 2*i + 2)
    plt.plot(D2D.dt*np.arange(MAX_SIM_STEPS), errors[:, i],'k')
    plt.ylabel('error in ' + labels[i])
plt.show()






