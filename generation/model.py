import numpy as np
import tensorflow as tf
import re
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ------------------------------
# CONFIGURATION
# ------------------------------
CORPUS_PATH = "/home/lokesh/Project/Python/rnn_assignment/generation/science_corpus.txt"
MODEL_PATH = "generation_rnn_model.keras"
TOKENIZER_PATH = "generation_tokenizer.pkl"
SEQ_LENGTH = 10
EMBED_DIM = 50
RNN_UNITS = 128
EPOCHS = 50
BATCH_SIZE = 32

# ------------------------------
# UTILITIES
# ------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s\.]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def generate_sequences(text, tokenizer, seq_length):
    tokens = tokenizer.texts_to_sequences([text])[0]
    sequences = []
    for i in range(seq_length, len(tokens)):
        sequences.append((tokens[i-seq_length:i], tokens[i]))
    X = np.array([x for x, _ in sequences])
    y = to_categorical([y for _, y in sequences], num_classes=len(tokenizer.word_index)+1)
    return X, y

def build_generation_model(vocab_size):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBED_DIM),
        LSTM(RNN_UNITS, return_sequences=True),
        Dropout(0.2),
        LSTM(RNN_UNITS),
        Dropout(0.2),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_text(seed_text, tokenizer, model, max_len, next_words=20, temperature=1.0):
    output = seed_text.lower()
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([output])[0][-max_len:]
        token_list = pad_sequences([token_list], maxlen=max_len)
        predictions = model.predict(token_list, verbose=0)[0]

        predictions = np.log(predictions + 1e-10) / temperature
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)

        predicted_index = np.random.choice(len(predictions), p=predictions)
        word = next((w for w, i in tokenizer.word_index.items() if i == predicted_index), '')
        output += f' {word}' if word else ''
    return output.strip()

# ------------------------------
# MAIN EXECUTION
# ------------------------------

with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
    raw_text = f.read()
text = clean_text(raw_text)

tokenizer = Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts([text])
vocab_size = len(tokenizer.word_index) + 1

X, y = generate_sequences(text, tokenizer, SEQ_LENGTH)

model = build_generation_model(vocab_size)
model.summary()

history = model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE)

model.save(MODEL_PATH)
with open(TOKENIZER_PATH, 'wb') as f:
    pickle.dump(tokenizer, f)

seed = "photosynthesis is the process by which"
generated = generate_text(seed, tokenizer, model, SEQ_LENGTH, next_words=20, temperature=0.5)

print(f"\nSeed: {seed}\nGenerated:\n{generated.upper()}")
