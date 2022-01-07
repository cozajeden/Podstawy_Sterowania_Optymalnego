import numpy as np

def zad4_1(a, b, c, d, e, bounds, precision=1):
    p = np.poly1d(np.array([a, b, c, d, e]))
    space = np.arange(bounds[0], bounds[1]+precision, step=precision)
    minmax = np.array([float('inf'), float('-inf')])
    for x in space:
        y = p(x)
        minmax[0] = min(minmax[0], y)
        minmax[1] = max(minmax[1], y)
    return minmax

minmax = zad4_1(1, 1, -129, 171, 1620, (-46, 14), precision=0.01)
print(minmax)