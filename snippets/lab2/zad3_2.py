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

x, y = signal.impulse(G1)
plt.plot(x, y)
plt.title('Impulse response')
plt.xlabel('time')