import numpy as np
import matplotlib.pyplot as plt

def zad5_2(*coefficients, bounds, precision=1):
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
    plt.annotate(
        f'x={x_min:.2f}\ny={y_min:.2f}',
        xy=(x_min, y_min),
        xytext=(x_min+2, y_min+100000)
    )
    plt.annotate(
        f'x={x_max:.2f}\ny={y_max:.2f}',
        xy=(x_max, y_max),
        xytext=(x_max+2, y_max-250000)
    )
    plt.legend(['polynomial', 'minimum', 'maximum'])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(''.join(
        ['{0}{1}x'.format('+' if c >= 0 else '',c)\
            for c in coefficients])
    )
    plt.tight_layout()
    plt.show()

zad5_2(1, 1, -129, 171, 1620, bounds=(-46, 14), precision=0.01)