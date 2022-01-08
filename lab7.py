from typing import Callable
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate.odepack import odeint
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are

# 4.1
RESOLUTION = 300
L = 1 #R
m = 9
J = 1
g = 10
d = 0.5

def A_of_x(x: np.ndarray) -> np.ndarray:
    if x[0, 0] == 0:
        return np.array([
            [0, 1],
            [0, -d/J]
        ]).astype(float)
    return np.array([
        [0, 1],
        [-m*g*L*np.sin(x[0, 0])/(J*x[0, 0]), -d/J]
    ]).astype(float)

B = np.array([[0, 1/J]]).T
R = np.array([[0.01]])
Q = np.array([
    [1, 0],
    [0, 1]
])

# 4.2
def riccati_finite_diff(
        x: np.ndarray,
        t: float,
        a: Callable,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray) -> np.ndarray:
    P = x[:4].reshape((2, 2))
    x = x[4:].reshape((2, 1))
    A = a(x)
    dP = - P@A - P@B@inv(R)@B.T@P - A.T@P + Q
    p = P@x
    u = -inv(R)@B.T@p
    dx = A@x + B@u
    return np.concatenate((dP.flatten(), dx.flatten()))

def riccati_infinite_diff(
        x: np.ndarray,
        t: float,
        a: Callable,
        B: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray) -> np.ndarray:
    x = x[4:].reshape((2, 1))
    A = a(x)
    P = solve_continuous_are(A, B, Q, R)
    u = -inv(R)@B.T@P@x
    dx = A@x + B@u
    return np.concatenate((P.flatten(), dx.flatten()))

def show_P(t: np.ndarray, res: np.ndarray) -> None:
    plt.plot(t, res[:, :4])
    plt.xlim(t[0], t[-1])
    plt.legend(['$p_{11}$', '$p_{12}$', '$p_{21}$', '$p_{22}$'])
    plt.show()
    plt.close()

x0 = [0]*2
P0 = [0]*4
t = np.linspace(5, 0, RESOLUTION)
res = odeint(
    riccati_finite_diff,
    P0 + x0,
    t,
    (A_of_x, B, Q, R))

show_P(t, res)

S = solve_continuous_are(
    A_of_x(
        np.array([0,0]).reshape((2, 1))
    ), B, Q, R
)

print('S =\n', S)
print('P =\n', res[-1, :4].reshape((2, 2)))

res = odeint(
    riccati_infinite_diff,
    P0 + x0,
    t,
    (A_of_x, B, Q, R))

show_P(t, res)

# 4.4
def show_x(t: np.ndarray, res: np.ndarray) -> None:
    plt.plot(t, res[:, 4:])
    plt.xlim(t[0], t[-1])
    plt.legend(['$x_1$', '$x_2$'])
    plt.show()
    plt.close()

t = np.linspace(0, 5, RESOLUTION)
x0 = [np.pi*2/3, 0]

res = odeint(
    riccati_finite_diff,
    P0 + x0,
    t,
    (A_of_x, B, Q, R))

show_x(t, res)

res = odeint(
    riccati_infinite_diff,
    P0 + x0,
    t,
    (A_of_x, B, Q, R))

show_x(t, res)