from typing import Callable
import numpy as np
from numpy.lib.function_base import append
from scipy import linalg, integrate
import matplotlib.pyplot as plt

R = 0.5
C = 0.5
L = 0.2

A = np.array([
    [0, 1],
    [-1/(L*C), -R/L]
])
B = np.array([
    [0], [1/L]
])

Q = np.eye(2)
R = np.array([[1]])

# 2.1
def calculateK(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray):
    P = linalg.solve_continuous_are(A, B, Q, R)
    return linalg.inv(R) @ B.T @ P

K = calculateK(A, B, Q, R)
print('K = ', K)

# 2.2
def model(
        x: np.ndarray,
        t: float,
        A: np.ndarray,
        B: np.ndarray,
        u: Callable):
    x = x.reshape((2, 1))
    dx = A@x + B@u(t)
    return dx.flatten()

# 2.3
t = np.linspace(0, 5, 1000)
y = integrate.odeint(
    model,
    [0, 0],
    t,
    (A, B, lambda x: np.array([[1]]))
)
plt.plot(t, y)
plt.legend(['$x_1$', '$x_2$'])
plt.show()
plt.close()

# 2.4
def modelK(
        x: np.ndarray,
        t: float,
        A: np.ndarray,
        B: np.ndarray,
        K: np.ndarray):
    x = x.reshape((2, 1))
    dx = A@x - B@K@x
    return dx.flatten()

# 2.5
def plotLQR(
        ax: plt.Axes,
        t: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        K: np.ndarray,
        title: str):
    y = integrate.odeint(
        modelK,
        [1, 1],
        t,
        (A, B, K)
    )
    ax.plot(t, y)
    ax.legend(['$x_1$', '$x_2$'])
    ax.set_title(title)

fig, ax = plt.subplots(1)
plotLQR(ax, t, A, B, K, f'K = {K}')
plt.show()
plt.close()

fig, axes = plt.subplots(3, 3)
fig.set_size_inches(10, 10)
for i, r in enumerate((2, 10, 30)):
    for j, q in enumerate((2, 10, 30)):
        k = calculateK(A, B, Q*q, R*r)
        plotLQR(axes[i][j], t, A, B, k, f'R = {R*r}, Q = {Q*q}')
plt.tight_layout()
plt.show()
plt.close()

# 2.6
def modelKJ(
        x: np.ndarray,
        t: float,
        A: np.ndarray,
        B: np.ndarray,
        K: np.ndarray,
        Q: np.ndarray,
        R: np.ndarray):
    j = x[1]
    x = x[1:].reshape((2, 1))
    u = -K@x
    j += x.T@Q@x + u.T@R@u
    dx = A@x + B@u
    return np.append(j.flatten(), dx.flatten())

y = integrate.odeint(
    modelKJ,
    [0, 1, 1],
    t,
    (A, B, K, Q, R)
)
plt.plot(t, y[:,1:])
plt.vlines(y[-1,:1], np.min(y[:,1:]), np.max(y[:,1:]), '#A0A0A0', '--')
plt.annotate(f'J = {y[-1, 0]}', (y[-1, 0], np.min(y[:,1:])))
plt.show()
plt.close()

# 3.1
def modelKe(
        x: np.ndarray,
        t: float,
        A: np.ndarray,
        B: np.ndarray,
        K: np.ndarray,
        qd: np.ndarray,
        Cap: float):
    qd_over_C = qd/Cap
    L = x[0]
    x = x[1:].reshape((2, 1))
    xd = np.array([[qd, 0]]).T
    e = xd - x
    ue = -K@e
    u = -ue+qd_over_C
    dx = A@x + B@u
    L += e.T@Q@e + ue.T@R@ue
    return np.append(L.flatten(), dx.flatten())

qd = 7
y = integrate.odeint(
    modelKe,
    [0, 1, 1],
    t,
    (A, B, K, qd, C)
)
plt.plot(t, y[:, 1:])
plt.show()
plt.close()

# 3.2
def plotLQR(
        ax: plt.Axes,
        t: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        K: np.ndarray,
        qd: float,
        C: float,
        title: str):
    y = integrate.odeint(
        modelKe,
        [0, 1, 1],
        t,
        (A, B, K, qd, C)
    )
    ax.plot(t, y[:, 1:])
    ax.legend(['$x_1$', '$x_2$'])
    ax.set_title(title)

fig, axes = plt.subplots(3, 3)
fig.set_size_inches(10, 10)
for i, r in enumerate((2, 10, 30)):
    for j, q in enumerate((2, 10, 30)):
        k = calculateK(A, B, Q*q, R*r)
        plotLQR(axes[i][j], t, A, B, k, qd, C, f'R = {R*r}, Q = {Q*q}')
plt.tight_layout()
plt.savefig()
plt.show()
plt.close()