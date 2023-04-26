import numpy as np
from CUR import CUR

A = np.linspace(0, 14, 15).reshape((3, -1))
print(A)
C, U, R, sigma = CUR(A, 3)
Q = np.dot(np.dot(C, U), R)
print('sigma=', sigma)
print('Q=', Q, Q.shape)
