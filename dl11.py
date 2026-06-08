from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

text = ("we will we will rock you " * 50).lower()

chars = sorted(set(text))
c2i = {c:i for i,c in enumerate(chars)}
i2c = {i:c for c,i in enumerate(chars)}

seq = 10
X, y = [], []

for i in range(len(text)-seq):
    X.append([c2i[c] for c in text[i:i+seq]])
    y.append(c2i[text[i+seq]])

X = to_categorical(X, len(chars))
y = to_categorical(y, len(chars))

model = Sequential([
    LSTM(64, input_shape=(seq, len(chars))),
    Dense(len(chars), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, y, epochs=5, verbose=0)

seed = text[:seq]
result = seed

for _ in range(100):
    x = to_categorical([[c2i[c] for c in result[-seq:]]], len(chars))
    pred = model.predict(x, verbose=0)
    next_char = i2c[np.argmax(pred)]
    result += next_char

print("Generated Lyrics:\n")
print(result)
