import numpy as np

def kronecker_delta(h, j):
    if h==j:
        return 1
    else:
        return 0
