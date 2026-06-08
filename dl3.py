import itertools

w = [0.2, 0.4, 0.2]
b = -0.5

def predict(x):
    return 1 if sum(i * j for i, j in zip(x, w)) + b >= 0 else 0

def actual(x):
    h, hn, c = x
    return 1 if hn and (h or c) else 0

correct = 0

print("Inputs (H,Hn,C) | Predicted | Actual")

for x in itertools.product([0, 1], repeat=3):
    p = predict(x)
    a = actual(x)

    if p == a:
        correct += 1

    print(x, "|", p, "|", a)

print("\nAccuracy:", (correct / 8) * 100, "%")
