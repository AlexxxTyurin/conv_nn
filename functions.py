import numpy as np
import pandas as pd


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), "constant", constant_values=0)
    return X_pad


# This function computes the result of a convolution over a slice
def conv_single_step(a_slice_prev, W, b):
    return np.sum(a_slice_prev * W) + b


# This function computes the result of a convolutional operation over the whole set
def conv_forward(A_prev, W, b, hparams):
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    f, f, n_C = W.shape

    stride = hparams['stride']
    padding = hparams['padding']

    n_H = int((n_H_prev - f + 2 * padding) / stride) + 1
    n_W = int((n_W_prev - f + 2 * padding) / stride) + 1

    Z = np.zeros([m, n_H, n_W, n_C])

    A_prev_pad = zero_pad(A_prev, padding)
    W = np.array([W for _ in range(m)])

    for h in range(n_H):
        for w in range(n_W):
            for c in range(n_C):
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f

                a_slice_prev = A_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end, :]
                Z[:, h, w, c] = conv_single_step(a_slice_prev, W, b)

    return Z


def pool(A_prev, hparams, mode='max'):
    m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape

    f = hparams["f"]
    stride = hparams["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for h in range(n_H):  # loop on the vertical axis of the output volume
        for w in range(n_W):  # loop on the horizontal axis of the output volume
            for c in range(n_C):  # loop over the channels of the output volume

                # Find the corners of the current "slice" (≈4 lines)
                vert_start = h * stride
                vert_end = vert_start + f
                horiz_start = w * stride
                horiz_end = horiz_start + f

                # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                a_prev_slice = A_prev[:, vert_start:vert_end, horiz_start:horiz_end, c]

                # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                if mode == "max":
                    A[:, h, w, c] = np.max(a_prev_slice)
                elif mode == "average":
                    A[:, h, w, c] = np.average(a_prev_slice)

    return A










