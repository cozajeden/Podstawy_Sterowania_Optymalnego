import numpy as np

b = np.array([[-1], [2]])
a = np.array([[1, 2], [-1, 0]])
x = np.linalg.solve(a, b)
print(x)