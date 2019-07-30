import numpy as np
import pandas as pd




def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    S = W * a_slice_prev + b
    Z = np.sum(S)
    return Z







