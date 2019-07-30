import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *


np.random.seed(1)

x = np.random.randn(4, 3, 3, 2)

x_pad = zero_pad(x, 2)

print(x_pad.shape)


