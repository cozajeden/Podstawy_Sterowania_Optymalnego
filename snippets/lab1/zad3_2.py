import numpy as np

precision = 0.01

a = np.array([1, 1, -129, 171, 1620])
p = np.poly1d(a)
space = np.arange(-46, 14+precision, step=precision)
minmax = np.array([float('inf'), float('-inf')])
for x in space:
    y = p(x)
    minmax[0] = min(minmax[0], y)
    minmax[1] = max(minmax[1], y)
print(minmax)