import numpy as np
from scipy import signal, integrate
from dataclasses import astuple, dataclass
import matplotlib.pyplot as plt

#1.2
@dataclass
class Model:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray

    def get(self):
        return astuple(self)

    @property
    def sys(self):
        return signal.StateSpace(*self.get())

C1 = 1
C2 = 0.5
R1 = 2
R2 = 4

rys1 = Model(
    np.array([
        [-1/(R1*C1), 0],
        [0, -1/(R2*C2)]
    ]),
    np.array([
        [1/(R1*C1)],
        [1/R2*C2]
    ]),
    np.array([[1, 0]]),
    np.array([[0]])
)

C1 = 1
C2 = 2
C3 = 3
R = 1

rys2 = Model(
    np.array([
        [-1/(R*C1), 0, 0],
        [0, -1/(R*C2), 0],
        [0, 0, -1/(R*C3)]
    ]),
    np.array([
        [1/(R*C1)],
        [1/(R*C2)],
        [1/(R*C3)]
    ]),
    np.array([[0, 0, 1]]),
    np.array([[0]])
)

C = 1
R = 1

rys3 = Model(
    np.array([[-1/(C*R)]]),
    np.array([[0]]),
    np.array([[1]]),
    np.array([[0]])
)

C1 = 2
R1 = 2
R2 = 1
L1 = 0.5
L2 = 1

rys4 = Model(
    np.array([
        [-R1/L1, 0, 1/L1],
        [0, 0, 1/L2],
        [-1/C1, -1/C1, -1/(C1*R2)]
    ]),
    np.array([
        [1/L1],
        [0],
        [0]
    ]),
    np.array([[0, 0, 1]]),
    np.array([[0]])
)

#1.3
def kalman(rys: Model):
    K = np.zeros_like(rys.A)
    for i in range(rys.A.shape[0]):
        K[:, i:i+1] = np.linalg.matrix_power(rys.A, i)@rys.B
    print(K)
    rank = np.linalg.matrix_rank(K)
    print(f'rank={rank}')
    print(f'Uklad{"" if rank==K.shape[0] else " nie"} sterowalny\n')

print('rys1:')
kalman(rys1)
print('rys2:')
kalman(rys2)
print('rys3:')
kalman(rys3)
print('rys4:')
kalman(rys4)

#1.4
def plot(rys:Model, ax:plt.Axes, u:np.ndarray, t:np.ndarray, title:str):
    t, y, x = signal.lsim2(rys.sys, u, t)
    ax.plot(t, y, linewidth=5, alpha=0.5)
    states = x.shape[1]
    ax.set_title(title)
    for i in range(states):
        ax.plot(t, x.T[i])
    ax.legend(['y'] + [f'$x_{s+1}$' for s in range(states)])

def sim(rys: Model, title: str):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(title)

    t = np.linspace(0, 20, 1000)
    u = np.ones((1000,))
    plot(rys, ax1, u, t, 'step response')
    
    t = np.linspace(0, 10*np.pi, 1000)
    u = np.sin(t)
    plot(rys, ax2, u, t, 'sinusoidal response')
    
    fig.tight_layout()
    plt.show()

sim(rys1, 'rysunek 1')
sim(rys2, 'rysunek 2')
sim(rys3, 'rysunek 3')
sim(rys4, 'rysunek 4')

#2.1
def to_controllable_canonical(rys: Model) -> Model:
    [num], den = signal.ss2tf(*rys.get())
    A = np.zeros_like(rys.A)
    A[:-1,1:] = np.eye(rys.A.shape[0] - 1)
    A[-1] = den[::-1][:-1]*-1
    B = np.zeros_like(rys.B)
    B[-1] = 1
    C = np.array([num[::-1][:-1]])
    return Model(A,B,C,np.array([[0]]))

rys2cc = to_controllable_canonical(rys2)
rys4cc = to_controllable_canonical(rys4)

print('rys2:\n', rys2cc)
print('rys4:\n', rys4cc)

#2.2
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
fig.suptitle('Rysunek 4')

t = np.linspace(0, 20, 1000)
u = np.ones((1000,))
plot(rys4, ax1, u, t, 'step response')
plot(rys4cc, ax2, u, t, 'step response - controllable form')

t = np.linspace(0, 5*np.pi, 1000)
u = np.sin(t)
plot(rys4, ax3, u, t, 'sinusoidal response')
plot(rys4cc, ax4, u, t, 'sinusoidal response - controllable form')

fig.tight_layout()
plt.show()