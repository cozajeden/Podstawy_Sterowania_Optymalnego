from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

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