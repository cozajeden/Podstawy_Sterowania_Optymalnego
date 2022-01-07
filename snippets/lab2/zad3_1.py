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

x, y = signal.impulse(G1)
plt.plot(x, y)
plt.title('Impulse response')
plt.xlabel('time')