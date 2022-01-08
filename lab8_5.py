from typing import Callable
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 5.2
RESOLUTION = 1000
A = 1.5
J = 1
R = 1
omega = 0.65
d = 0.5
m = 1
g = 10

def tau_m(t:float, omega:float, A:float) -> float:
    return A*np.sin(omega*t)

def model(
    x: np.ndarray,
    t: float,
    d: float,
    m: float,
    g: float,
    omega: float,
    A: float,
    J: float,
    R: float,
    tau_m: Callable[[float, float, float], float]
) -> np.ndarray:
    x1, x2 = x
    dx1 = x2
    dx2 = (tau_m(t, omega, A) - d*x2 - m*g*R*np.sin(x1))/J
    return np.array([dx1, dx2])

# 5.3
t = np.linspace(0, 30, RESOLUTION)

res = odeint(
    model,
    np.array([0, 0]),
    t,
    (d, m, g, omega, A, J, R, tau_m)
)

plt.plot(t, res)
plt.xlabel('t')
plt.title('x(t)')
plt.legend(['$x_1$', '$x_2$'])
plt.show()
plt.close()