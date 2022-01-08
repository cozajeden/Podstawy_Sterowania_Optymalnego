# 2.1
from typing import List
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 2.2
RESOLUTION = 100

def model(y: np.ndarray, t:float) -> List:
    return [t**2]

# 2.3
t = np.linspace(0, 10, RESOLUTION)

res_n = odeint(model, [0], t)

# 2.4
res_a = t**3/3

plt.plot(t, res_n, linewidth=5, alpha=0.5)
plt.plot(t, res_a)
plt.title("comparison of numerical and analytical solution")
plt.legend(['numeric', 'analytical'])
plt.xlabel("t")
plt.show()
plt.close()
# Wykresy się pokrywają
# Zostało użyte całkowanie układu równań różniczkowych zwyczajnych