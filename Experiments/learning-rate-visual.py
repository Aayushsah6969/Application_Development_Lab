import numpy as np
import matplotlib.pyplot as plt

# Simple convex cost function: J(w) = w^2
def cost(w):
    return w**2

def gradient(w):
    return 2*w

def gradient_descent(lr, steps=25, w0=8):
    w = w0
    costs = []
    for _ in range(steps):
        costs.append(cost(w))
        w = w - lr * gradient(w)
    return costs

# Learning rates
lr_low = 0.02
lr_good = 0.2
lr_high = 1.1

c_low = gradient_descent(lr_low)
c_good = gradient_descent(lr_good)
c_high = gradient_descent(lr_high)

# Plot
plt.figure()
plt.plot(c_low, label="Too Low LR")
plt.plot(c_good, label="Moderate LR")
plt.plot(c_high, label="Too High LR")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Effect of Learning Rate on Gradient Descent")
plt.legend()
plt.grid(True)
plt.show()
