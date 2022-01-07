def u(t):
    return np.array([1.0])

def model(t, y, A, B, u):
    dy = A@y + B@u(t)
    return dy