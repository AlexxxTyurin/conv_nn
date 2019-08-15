import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *

data = np.array(pd.read_csv("/Users/alextyurin/Desktop/pycharm_projects/recognition/train.csv"))

Y = np.array(data[:, 0])

X = np.array(data[:, 1:])
m = X.shape[0]
X = np.reshape(X, [m, 28, 28, 1])
W = np.random.randn(28, 28, 3, 1)
print(X.shape)

hparams = {'stride': 1, 'padding': 2}

X = zero_pad(X, 2)
print(X.shape)
# print(conv_forward(X, W, 0, hparams))





