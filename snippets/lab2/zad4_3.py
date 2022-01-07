t_max = 100
t = np.linspace(0, t_max, 1000)
u = t/t_max
x, y, _ = signal.lsim2(sys, u, t)
plt.plot(x, y)
plt.title('rising $\\tau_m$')
plt.xlabel('time')
plt.ylabel('$\Theta$')

x, y, _ = signal.lsim2(sys, u[::-1], t)
plt.plot(x, y)
plt.title('falling $\\tau_m$')
plt.xlabel('time')
plt.ylabel('$\Theta$')