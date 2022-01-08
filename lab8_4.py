import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 4.1
RESOLUTION = 100
kp = 2
T = 2
kob = 4
x = 1

# 4.2
def feedback(
    y:np.ndarray,
    t:float,
    kp:float,
    T:float,
    kob:float,
    x:float
) -> np.ndarray:
    u = kp*(x - y)
    u_clip = np.clip(u, -0.1, 0.1)
    dy = kob*u_clip - T*y
    return dy

# 4.3
t = np.linspace(0, 5, RESOLUTION)
res = odeint(feedback, [0], t, args=(kp, T, kob, x))

plt.plot(t, res)
plt.title("Feedback")
plt.xlabel("t")
plt.ylabel("y")
plt.show()
plt.close()

# 4.4
for x in (1, 2, 3):
    res = odeint(feedback, [0], t, args=(kp, T, kob, x))
    plt.plot(t, res)

plt.title("Feedback for different x")
plt.xlabel("t")
plt.ylabel("y")
plt.legend(["$x=1$", "$x=2$", "$x=3$"])
plt.show()
plt.close()

# 4.5
def feedback(
    y:np.ndarray,
    t:float,
    kp:float,
    T:float,
    kob:float,
    x:float
) -> np.ndarray:
    u = kp*(x - y)
    dy = kob*u - T*y
    return dy

for x in (1, 2, 3):
    res = odeint(feedback, [0], t, args=(kp, T, kob, x))
    plt.plot(t, res)

plt.title("Feedback for different x")
plt.xlabel("t")
plt.ylabel("y")
plt.legend(["$x=1$", "$x=2$", "$x=3$"])
plt.show()
plt.close()