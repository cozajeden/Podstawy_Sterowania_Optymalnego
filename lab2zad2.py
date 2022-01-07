#2.1
from scipy import integrate, signal
import matplotlib.pyplot as plt
import numpy as np

# 2.6
def u(t):
    return np.array([1.0])

def model(t, y, A, B, u):
    dy = A@y + B@u(t)
    return dy

if __name__ == "__main__":
    # 2.2
    kp = 3
    T = 2
    A = np.array([-1/T])
    B = np.array([kp/T])
    C = np.array([1])
    D = np.array([0])
    # 2.3
    G1 = signal.TransferFunction([kp], [T, 1])
    # 2.4
    x, y = signal.step(G1)
    plt.plot(x, y)
    plt.xlabel('time')
    plt.savefig('images/lab2/zad2_4.jpg')
    plt.close()
    # 2.5
    sys = signal.StateSpace(A, B, C, D)
    x, y = signal.step(sys)
    plt.plot(x, y)
    plt.xlabel('time')
    plt.savefig('images/lab2/zad2_5.jpg')
    plt.close()
    # 2.7
    t_eval = np.linspace(0,15,100)
    # 2.8
    result = integrate.solve_ivp(model, [0,15], [0], t_eval=t_eval, args=(A, B, u,))
    # 2.9
    plt.plot(result.t, result.y[0])
    plt.xlabel('time')
    plt.savefig('images/lab2/zad2_9.jpg')
    plt.close()