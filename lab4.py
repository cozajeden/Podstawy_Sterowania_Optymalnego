# 5.1
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


# 5.2
Kp = 2
Ti = 1
Td = 0.4

lti = signal.lti([], [-3]*3, 5)
Go = signal.TransferFunction(lti)
A = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [-1, -3, -3]
])
B = np.array([[1], [0], [0]])
C = np.array([[5]])
D = np.array([[0]])

# 5.3
def closed_loop_with_PID(Kp: float, Td: float, Ti: float) -> signal.TransferFunction:
    G = signal.TransferFunction(
        [5*Kp*Td, 5*Kp, 5*Kp/Ti],
        [1, 9, 5*Kp*Td+27, 5*Kp+27, 5*Kp/Ti]
    )
    return G

Gz = closed_loop_with_PID(Kp, Td, Ti)

# 5.4
x, y = signal.step(Gz)
plt.plot(x, y)
plt.xlabel('time')
plt.title('Odpowiedź układu zamkniętego\nz regulatorem PID')
plt.show()
plt.close()

# 5.5
sysz = Gz.to_ss()
print('A =', sysz.A)
print('B =', sysz.B)
print('C =', sysz.C)
print('D =', sysz.D)

# 6.1 - 6.4
t = np.linspace(0, 15, 1000)
x, y = signal.step(Go.to_ss(), T=t)

plt.plot(x, y)
plt.show()
plt.close()

# Wyznaczenie wartości K
K = np.max(y)

# Obliczanie punktu przegięcia
dy = y[1:] - y[:-1]
ddy = dy[1:] - dy[:-1]
ddy_zero = np.isclose(ddy, np.zeros_like(ddy), atol=1e-5)
x_ddy_zero_lok = np.where(ddy_zero == True)[0][0]
print(x_ddy_zero_lok)
x_inflect = x[x_ddy_zero_lok: x_ddy_zero_lok + 2]
y_inflect = y[x_ddy_zero_lok: x_ddy_zero_lok + 2]

# wyznaczanie stycznej do wykresu w punkcie przegięcia
tangent = np.poly1d(np.polyfit(x_inflect, y_inflect, 1))
y_tangent = tangent(x)

# Wyznaczenie wartości T0
T0_where = np.where(y_tangent >= 0)[0][0]
T0 = x[T0_where]

# Wyznaczenie wartości T
T_where = np.where(y_tangent <= K)[0][-1]
T_x = x[T_where]
T = T_x - T0

plt.plot(x, y)
plt.plot(x[T0_where:T_where+1], y_tangent[T0_where:T_where+1], '--')
plt.plot(T0, 0, '.')
plt.plot(T_x, 0, '.')
plt.hlines(K, x[0], x[-1], '#A0A0A0', '--')
plt.vlines(T_x, y[0], y[-1], '#A0A0A0', '--')
plt.annotate('K', (0, K))
plt.annotate('$T_0$', (T0, 0))
plt.annotate('$T_0+T$', (T_x, 0))
plt.title('Step response')
plt.xlabel('time')
plt.legend(['response', 'tangent'])
plt.show()
plt.close()

# 6.4
Kp = 1.2*T/(K*T0)
Ti = 2*T0
Td = 0.4*T0
print('T0/T = ', T0/T)
print('Kp = ', Kp)
print('Ti = ', Ti)
print('Td = ', Td)

# 6.5
syszn = closed_loop_with_PID(Kp, Td, Ti).to_ss()
x, y = signal.step(syszn)
plt.plot(x, y)
plt.xlabel('time')
plt.title('Odpowiedź układu zamkniętego\nzoptymalizowanego metodą ZN')
plt.show()
plt.close()

# 6.6
e = np.ones_like(y) - y
print('I_IAE = ', np.sum(e))


class Experiment:

    def __init__(self, Kp: float, Td: float, Ti: float) -> None:
        self.Kp = Kp
        self.Td = Td
        self.Ti = Ti
        x, y = signal.step(self.sys)
        self.e = self.error(x, y)

    @property
    def sys(self) -> signal.StateSpace:
        return closed_loop_with_PID(
            self.Kp, self.Td, self.Ti
        ).to_ss()

    def error(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(np.abs(np.ones_like(y) - y))

    def __str__(self) -> str:
        return f"Kp={self.Kp:.3f}, Ti={self.Ti:.3f}, Td={self.Td:.3f}, e={self.e:.3f}."


experiments = []

for Kp in np.linspace(0.001, 50, 20):
    for Ti in np.linspace(0.001, 5, 20):
        for Td in np.linspace(0.001, 5, 20):
            experiments.append(Experiment(Kp, Td, Ti))

sorted_exp = sorted(experiments, key=lambda x: x.e)

print('Wartości dla najniższego znalezionego experymentalnie\
kryterium IAE to: ', sorted_exp[0])
plt.plot(*signal.step(sorted_exp[0].sys))
plt.title('Odpowiedź skokowa dla experymentalnie\n\
znalezionych wartości PID minimalizując IAE')
plt.show()
plt.close()

# 6.7
# Poszukiwanie K krytycznego
slope = np.inf
Kkr = 0
for k in np.linspace(30, 50, 1000):
    ss = closed_loop_with_PID(k, 0, np.inf).to_ss()
    x, y = signal.step(ss)
    peaks_pos = signal.find_peaks(y)[0]
    peaks = y[peaks_pos]
    sl = np.sum(np.abs(peaks[1:] - peaks[:-1]))
    if sl < slope:
        slope = sl
        Kkr = k
print('K krytyczne = ', Kkr)
sysKkr = closed_loop_with_PID(Kkr, 0, np.inf).to_ss()

# Wyznaczenie okresu oscylacji
x, y = signal.step(sysKkr)
peaks_pos = signal.find_peaks(y)[0]
Tosc = x[peaks_pos[3]] - x[peaks_pos[2]]
print('Tosc = ', Tosc)

# Wyznaczenie nastaw regulatora PID
Kp = 0.6*Kkr
Ti = 0.5*Tosc
Td = Tosc/8
print(f'Wyznaczone parametry drugą metodą \
Zieglera – Nicholsa to: Kp={Kp:.3f}, Ti={Ti:.3f}, Td={Td:.3f}')

# 6.8
syszn2 = closed_loop_with_PID(Kp, Td, Ti).to_ss()
x, y = signal.step(syszn2)
plt.plot(x, y)
plt.title('Odpowiedź skokowa układu z PID\nwyznaczonym druga metodą ZN')
plt.show()
plt.close()

# 6.10
class Experiment2(Experiment):

    def error(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(x*((np.ones_like(y) - y)**2))


zn2_itse = Experiment2(Kp, Td, Ti)
print('Nastawy PID oraz wartość kryterium ITSE \
dla drugiej metody Zieglera-Nicholsa to: ', zn2_itse)

experiments = []
for Kp in np.linspace(15, 35, 20):
    for Ti in np.linspace(0.001, 3, 20):
        for Td in np.linspace(0.001, 3, 20):
            experiments.append(Experiment2(Kp, Ti, Td))

sorted_exp = sorted(experiments, key=lambda x: x.e)
print('Znalezione experymentalnie nastawy PID minimalizujące\
ITSE to: ', sorted_exp[0])
x, y = signal.step(sorted_exp[0].sys)
plt.plot(x, y)
plt.title('Odpowiedź skokowa dla experymentalnie\
\nznalezionych wartości PID minimalizując ITSE')
plt.show()
plt.close()

# 7.1
print(f'R = {T/T0:.3f}')


class Experiment3(Experiment):

    def error(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.sum((np.ones_like(y)-y)**2)


exp_a = Experiment3(0.3*T/(K*T0), 0.5*T0, T)
exp_20 = Experiment3( 0.7*T/(K*T0), 4.7*T0, 1.4*T)

# 7.2
res_a = signal.step(exp_a.sys)
res_20 = signal.step(exp_20.sys)
plt.plot(*res_a, *res_20)
plt.title('Odpowiedzi skokowe przy regulacji metodą CHR')
plt.legend(['aperiodyczna', '20% przeregulowania'])
plt.show()
plt.close()

#7.3
print('Wartości dla regulatora CHR aperiodycznego, \
oraz błąd ISE to: ', exp_a)
print('Wartości dla regulatora CHR z przeregulowaniem 20%, \
oraz błąd ISE to: ', exp_20)

experiments = []
for Kp in np.linspace(2, 25, 20):
    for Ti in np.linspace(0.5, 3, 20):
        for Td in np.linspace(0.001, 3, 20):
            experiments.append(Experiment3(Kp, Ti, Td))

sorted_exp = sorted(experiments, key=lambda x: x.e)
print('Znalezione experymentalnie nastawy PID \
minimalizujące ISE to: ', sorted_exp[0])
x, y = signal.step(sorted_exp[0].sys)
plt.plot(x, y)
plt.title('Odpowiedź skokowa dla experymentalnie\
\nznalezionych wartości PID minimalizując ISE')
plt.show()
plt.close()