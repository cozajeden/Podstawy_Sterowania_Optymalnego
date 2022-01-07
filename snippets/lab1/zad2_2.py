import numpy as np

a = np.array([1, 1, -129, 171, 1620])
p = np.poly1d(a)
print('dla x=-46, y={0}'.format(p(-46)))
print('dla x=14, y={0}'.format(p(14)))