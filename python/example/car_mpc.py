#!/usr/bin/env python3
from dyno_model.car_ltv_mpc import *
from optimizer.qp import *
from cost.car_mpc import *
import matplotlib.pyplot as plt

############### TUNING PARAMS: ##########################
xmin = None
xmax = None
" u = [accel, steering]"
umin = [-5., -np.pi/4.0]
umax = [10., np.pi/4.0]

MAX_SIM_STEPS = 1000
TARGET_THRESHOLD = 0.2

Q = np.array([[200, 0], [0, 10.]]) # 3x3 in paper need to check
R = 5e4* np.eye(2)
N = 5
Nc = 1
mu = 0.3
T = 0.05
rho = 1e3
#########################################################

# Using parameters in Predictive Active Steering Control for Autonomous Vehicle Systems;
# https://borrelli.me.berkeley.edu/pdfpub/pub-2.pdf
D2D = Driver2DMPC(N = N, Nc = Nc, dt = T, mu = mu, rho = rho, \
                            umin = umin, umax= umax, xmin = xmin,\
                                    xmax = xmax, Q=Q, R=R)
Cost = Driver2DCost(D2D)
QP = QP()

u_optimal = np.array([10.0, 0.0])

state_new_ode = np.random.multivariate_normal(mean = [0.0]*6, cov = 1.*np.eye(6), size = 1).flatten().tolist()

state_new_ode[0] = 0.0
state_new_ode[1] = 0.0

state_new_linear = state_new_ode
D2D.state = state_new_ode

start_state = state_new_linear
del_u = np.zeros((2, 6))

target = []
state = []
ctrl=[]

for i in range(MAX_SIM_STEPS):

    D2D.ym = [2., 2.]
    D2D.state = state_new_ode
    state_new_linear = D2D.predict(u_optimal)

    state_new_ode = D2D.predict_ode(u_optimal, [0.0, 0.0], D2D.dt)

    del_u = D2D.get_optimal_control(QP, state_new_ode, u_optimal)

    print("u_optimal : ", del_u)

    state+=[D2D.state]
    target+= [D2D.ym]
    ctrl+=[del_u]

state = np.reshape(state, (-1, 6))
target = np.reshape(target, (-1, 2))
ctrl = np.reshape(ctrl, (-1, 2))

plt.figure()

plt.subplot(1, 2, 1)
plt.plot(state[:, 0], state[:, 1], 'k', label = 'trajectory')
plt.plot(target[:, 0], target[:, 1], 'or', label = 'target')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ctrl[:, 0], 'b', label = 'acceleration')
plt.plot(ctrl[:, 1], 'k', label = 'steering')
plt.legend()
plt.show()



#labels = ['$x$', '$y$', '$\dot{x}$', '$\dot{y}$', '$\psi$', '$\dot{\psi}$']
#
#plt.figure()
#plt.title('Linearized States vs. ODE States at dt = ' + str(D2D.dt) + ' sec')
#
#for i in range(6):
#    plt.subplot(6, 2, 2*i + 1)
#    plt.plot(D2D.dt*np.arange(MAX_SIM_STEPS), ode[:, i], label='ode')
#    plt.plot(D2D.dt*np.arange(MAX_SIM_STEPS), linear[:, i], label='linear')
#    plt.ylabel(labels[i])
#    max_ = np.max(ode[:, i])
#    min_ = np.min(ode[:, i])
#    plt.ylim((min_ - 10, max_ + 10))
#    plt.legend()
#
#for i in range(6):
#    plt.subplot(6, 2, 2*i + 2)
#    plt.plot(D2D.dt*np.arange(MAX_SIM_STEPS), errors[:, i],'k')
#    plt.ylabel('error in ' + labels[i])
#plt.show()






