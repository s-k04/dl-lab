import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

pairs = {
    "hello": "hi there",
    "how are you": "i am fine",
    "what is your name": "i am a chatbot",
    "bye": "goodbye",
    "thank you": "you are welcome"
}

questions = list(pairs.keys())
answers = list(pairs.values())

tokenizer = Tokenizer()
tokenizer.fit_on_texts(questions)

X = tokenizer.texts_to_sequences(questions)
X = pad_sequences(X, maxlen=5)

y = to_categorical(range(len(answers)), num_classes=len(answers))

model = Sequential([
    Embedding(len(tokenizer.word_index) + 1, 16, input_length=5),
    Bidirectional(LSTM(32)),
    Dense(len(answers), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X, y, epochs=100, verbose=0)

print("Chatbot Ready! Type 'exit' to stop.")

while True:
    user = input("You: ").lower()

    if user == "exit":
        break

    seq = tokenizer.texts_to_sequences([user])
    seq = pad_sequences(seq, maxlen=5)

    pred = np.argmax(model.predict(seq, verbose=0))

    print("Bot:", answers[pred])
