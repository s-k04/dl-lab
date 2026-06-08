import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

def softmax(x):
    e = np.exp(x - np.max(x))   # numerically stable
    return e / np.sum(e)

funcs = [sigmoid, tanh, relu, leaky_relu, softmax]
names = ["Sigmoid", "Tanh", "ReLU", "Leaky ReLU", "Softmax"]

for f, n in zip(funcs, names):
    plt.figure(figsize=(6, 4))
    plt.plot(x, f(x))
    plt.title(n)
    plt.grid(True)
    plt.show()ne