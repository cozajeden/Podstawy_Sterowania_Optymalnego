from scipy import signal
from scipy.signal import tf2ss, ss2tf
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

IMAGE_PATH = 'images/lab2'
SNIP_PATH = 'snippets/lab2'

if __name__ == "__main__":
    #3.1
    R = 12
    L = 1
    Cap = 0.000_1
    num = [1, 0]
    den = [L, R, 1/Cap]
    G1 = signal.TransferFunction(num, den)
    x, y = signal.step(G1)
    plt.plot(x, y)
    plt.title('Step response')
    plt.xlabel('time')
    plt.savefig(join(IMAGE_PATH, 'zad3_1step.jpg'))
    plt.close()
    x, y = signal.impulse(G1)
    plt.plot(x, y)
    plt.title('Impulse response')
    plt.xlabel('time')
    plt.savefig(join(IMAGE_PATH, 'zad3_1impulse.jpg'))
    plt.close()

    #3.2
    A = np.array([
        [0, 1],
        [-1/(L*Cap), -R/L]
    ])
    B = np.array([[0], [1/L]])
    C = np.array([0, 1])
    D = np.array([0])
    sys1 = signal.StateSpace(A, B, C, D)
    x, y = signal.step(sys1)
    plt.plot(x, y)
    plt.title('Step response')
    plt.xlabel('time')
    plt.savefig(join(IMAGE_PATH, 'zad3_2step.jpg'))
    plt.close()
    x, y = signal.impulse(G1)
    plt.plot(x, y)
    plt.title('Impulse response')
    plt.xlabel('time')
    plt.savefig(join(IMAGE_PATH, 'zad3_2impulse.jpg'))
    plt.close()

    #3.3
    G2 = ss2tf(A, B, C, D)
    sys2 = tf2ss(num, den)
    print(f'G1 = {G1}\n')
    print(f'G2 = {G2}\n')
    print(f'sys1 = {sys1}\n')
    print(f'sys2 = {sys2}\n')

    output = f'''G1 = {G1}\n
G2 = {G2}\n
sys1 = {sys1}\n
sys2 = {sys2}'''

    with open(join(SNIP_PATH, 'zad3_3out.txt'), 'w') as f:
        f.write(output)

    #3.4
    L = 0.15
    den = [L, R, 1/Cap]
    A = np.array([
        [0, 1],
        [-1/(L*Cap), -R/L]
    ])
    B = np.array([[0], [1/L]])
    G1 = signal.TransferFunction(num, den)
    sys1 = signal.StateSpace(A, B, C, D)
    G2 = ss2tf(A, B, C, D)
    sys2 = tf2ss(num, den)
    print(f'G1 = {G1}\n')
    print(f'G2 = {G2}\n')
    print(f'sys1 = {sys1}\n')
    print(f'sys2 = {sys2}\n')

    output = f'''G1 = {G1}\n
G2 = {G2}\n
sys1 = {sys1}\n
sys2 = {sys2}'''

    with open(join(SNIP_PATH, 'zad3_4out.txt'), 'w') as f:
        f.write(output)