# 3.1
from typing import Tuple
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 3.2
RESOLUTION = 100

kp = 2
omega = 4
zeta = 0.25
u = 1

# 3.3
def model(
    x:np.ndarray,
    t:float,
    kp:float,
    omega:float,
    zeta:float,
    u:float
) -> Tuple[float, float]:
    x1, x2 = x
    dxdt1 = x2
    dxdt2 = -(2*zeta*x2 + np.sqrt(x1))/omega + kp*u/omega**2
    return dxdt1, dxdt2

t = np.linspace(0, 50, RESOLUTION)

res = odeint(model, (0, 0), t, args=(kp, omega, zeta, u))

plt.plot(t, res)
plt.title('step response')
plt.legend(['$x_1$', '$x_2$'])
plt.xlabel("t")
plt.show()
plt.close()
# Odpowiedź układu ma charakter liniowych gasnących oscylacji