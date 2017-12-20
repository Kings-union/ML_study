import numpy as np
import func_regression

x, y = func_regression.genData(100, 25, 10)
# print(x, y)

m, n = np.shape(x)
n_y = np.shape(y)

# print(str(m), n, str(n_y))

numIterations = 100000
alpha = 0.0005
theta = np.ones(n)
print(theta)
theta = func_regression.gradientDescent(x, y, theta, alpha, m, numIterations)
print(theta)