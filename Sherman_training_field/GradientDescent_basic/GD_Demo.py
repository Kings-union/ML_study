import numpy as np
import matplotlib.pyplot as plt

x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208., 606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]
# ydata = b + w * xdata


b = -120    # initial b
w = -4  # initial w
lr = 0.00000001  # learning rate
iteration = 100000

# store initial values for plotting.
b_history = [b]
w_history = [w]

# Iterations
for i in range(iteration):
    b_grad = 0.0
    w_grad = 0.0
    for n in range(len(x_data)):
        b_grad = b_grad - 2.0*(y_data[n] - b - w*x_data[n])*1.0
        w_grad = w_grad - 2.0*(y_data[n] - b - w*x_data[n])*x_data[n]
    # update parameters.
    b = b - lr*b_grad
    w = w - lr*w_grad

    # store parameters for plotting
    b_history.append(b)
    w_history.append(w)

plt.plot(b_history, w_history, "o-", ms=3, lw=1.5, color="blue")
plt.xlabel(r'$b$', fontsize=16)
plt.ylabel(r'$w$', fontsize=16)
plt.show()
