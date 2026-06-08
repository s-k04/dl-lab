import numpy as np

def hebbian(x, y, w, lr=0.1):
    return w + lr * np.outer(x, y)

def perceptron(x, y, w, lr=0.1):
    return w + lr * x * y

def delta(x, y, w, lr=0.1):
    return w + lr * (y - np.dot(w, x)) * x

def correlation(x, y, w, lr=0.1):
    return w + lr * np.outer(x, y)

def outstar(x, y, w, lr=0.1):
    return w + lr * (y - np.dot(w, x))

x = np.array([1, -1, 0, 0.5])
y = 1
w = np.array([0.2, -0.1, 0.0, 0.1])

print("Hebbian:", hebbian(x, y, w))
print("Perceptron:", perceptron(x, y, w))
print("Delta:", delta(x, y, w))
print("Correlation:", correlation(x, y, w))
print("OutStar:", outstar(x, [y], w))