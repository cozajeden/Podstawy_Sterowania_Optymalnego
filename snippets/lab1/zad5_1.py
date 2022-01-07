import numpy as np
import matplotlib.pyplot as plt

def zad5_1(*coefficients, bounds, precision=1):
    p = np.poly1d(np.array(coefficients))
    space = np.arange(bounds[0], bounds[1]+precision, precision)
    y_min = float('inf')
    x_min = float('inf')
    y_max = float('-inf')
    x_max = float('-inf')
    y = np.array([])
    for x in space:
        temp = p(x)
        y = np.append(y, [temp])
        if temp < y_min:
            y_min = temp
            x_min = x
        if temp > y_max:
            y_max = temp
            x_max = x
    plt.plot(space, y)
    plt.plot([x_min], [y_min], 'o')
    plt.plot([x_max], [y_max], 'o')
    plt.tight_layout()
    plt.show()

zad5_1(1, 1, -129, 171, 1620, bounds=(-46, 14), precision=0.01)