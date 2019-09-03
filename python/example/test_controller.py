#!/usr/bin/env python3.5
from dyno_model.neural_network_predictor import *
from optimizer.newton_raphson import *

filename= "../model_data/neural_network_2.hdf5"

NNP = NeuralNetworkPredictor(model_file = filename, N1 = 5, N2= 10, Nu = 5, ym = [0.0]*5, K = 5, yn = [1.]*5, lambd = [0.3]*5)
NNP.compute_hessian(100, [0.5]*5, [1.]*5)

NNP.compute_cost([0.5]*5, [1.]*5)
NNP.compute_jacobian(100, [0.5]*5, [1.]*5)
NR_opt = NewtonRaphson(cost = NNP.Cost, d_model = NNP)
NR_opt.optimize(100, [0.5]*5, [1.]*5)


