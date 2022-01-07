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