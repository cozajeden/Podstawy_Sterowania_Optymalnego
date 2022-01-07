import numpy as np

x = np.array([
    [1, -2, 0],
    [-2, 4, 0],
    [2, -1, 7]])
    
x = np.linalg.matrix_rank(x)
print(x)