#!/usr/bin/env python3.5
from dyno_model.neural_network_predictor import *

filename= "../model_data/neural_network_2.hdf5"


NNP = NeuralNetworkPredictor(model_file = filename, N1 = 5, N2= 10, Nu = 5, ym = [0.0, 0.0], K = 5, yn = [1., 1.], lambd = [0.3]*5)
NNP.compute_hessian(100, [0.5]*10, [1.]*10)

