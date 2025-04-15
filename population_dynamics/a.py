import numpy as np

a = np.zeros((10, 10))
a[0, :] = np.arange(1, 11)
a[1:, -1] = np.arange(11, 20)
print(a[1:, -1])
print(a)