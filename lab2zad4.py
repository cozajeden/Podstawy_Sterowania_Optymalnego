from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from os.path import join

IMAGE_PATH = 'images/lab2'

# 4.1
m = 1
L = 0.5
d = 0.1
J = m*L**2/3
A = np.array([
    [0, 1],
    [0, -d/J]
])
B = np.array([
    [0],
    [1/J]
])
C = np.array([
    [1, 0]
])
D = np.array([[0]])
sys = signal.StateSpace(A, B, C, D)

#4.2
x, y = signal.step(sys)
plt.plot(x, y)
plt.title('Step response')
plt.xlabel('time')
plt.ylabel('$\Theta$')
plt.savefig(join(IMAGE_PATH, 'zad4_2.jpg'))
plt.close()

#4.3
t_max = 100
t = np.linspace(0, t_max, 1000)
u = t/t_max
x, y, _ = signal.lsim2(sys, u, t)
plt.plot(x, y)
plt.title('rising $\\tau_m$')
plt.xlabel('time')
plt.ylabel('$\Theta$')
plt.savefig(join(IMAGE_PATH, 'zad4_3_rising.jpg'))
plt.close()
x, y, _ = signal.lsim2(sys, u[::-1], t)
plt.plot(x, y)
plt.title('falling $\\tau_m$')
plt.xlabel('time')
plt.ylabel('$\Theta$')
plt.savefig(join(IMAGE_PATH, 'zad4_3_falling.jpg'))
plt.close()

#4.4 proporcjonalno całkujący
w, mag, phase = signal.bode(sys)
plt.subplot(2, 1, 1)
plt.semilogx(w, mag)
plt.title('Magnitude')
plt.ylabel('Magnitude, [dB]')
plt.xlabel('$\omega[rad/s]$')
plt.subplot(2, 1, 2)
plt.semilogx(w, phase)
plt.title('Phase')
plt.ylabel('Phase, [$\circ$]')
plt.xlabel('$\omega[rad/s]$')
plt.tight_layout()
plt.savefig(join(IMAGE_PATH, 'zad4_4.jpg'))
plt.close()