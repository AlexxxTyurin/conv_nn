import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *

data = np.array(pd.read_csv("/Users/vladikturin/Desktop/pycharm_projects/recognition/train.csv"))

Y = np.array(data[:, 0])

X = np.array(data[:, 1:])
m = X.shape[0]
X = np.reshape(X, [m, 28, 28, 1])
W = np.random.randn(5, 5, 1)

print(W.shape)

hparams = {'stride': 1, 'padding': 2}
result = conv_forward(X, W, 0, hparams)
print(result.shape)

result = pool(result, {'f': 2, 'stride': 2})
print(result.shape)





