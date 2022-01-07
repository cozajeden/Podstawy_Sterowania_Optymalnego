sys = signal.StateSpace(A, B, C, D)
x, y = signal.step(sys)
plt.plot(x, y)
plt.xlabel('time')