# model.py - RNN Text Classification (Math, Science, History)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout

# ------------------------------
# CONFIGURATION
# ------------------------------
DATA_PATH = "/home/lokesh/Project/Python/rnn_assignment/classification/classification_dataset.csv"
MODEL_PATH = "classification_rnn.keras"
LABEL_MAP_PATH = "label_mapping.txt"
MAX_EPOCHS = 20
BATCH_SIZE = 16

# ------------------------------
# FUNCTIONS
# ------------------------------

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df['Text'].values, df['Label'].values

def preprocess_texts(texts, labels):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_len)

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_onehot = to_categorical(labels_encoded)

    return padded_sequences, labels_onehot, tokenizer, label_encoder, max_len

def build_model(vocab_size, num_classes):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32),
        SimpleRNN(32),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_history(history, filename="training_history.png"):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)

def save_label_map(label_encoder, filepath):
    with open(filepath, "w") as f:
        for i, label in enumerate(label_encoder.classes_):
            f.write(f"{i}: {label}\n")

def demonstrate(model, tokenizer, label_encoder, max_len):
    samples = [
        "The study of chemical reactions and compounds.",
        "Solving for x in an equation like 2x + 5 = 11.",
        "What is the force that keeps planets in orbit?",
        "The period of rebirth in European art and literature."
    ]
    sequences = tokenizer.texts_to_sequences(samples)
    padded = pad_sequences(sequences, maxlen=max_len)
    preds = model.predict(padded)
    for text, pred in zip(samples, preds):
        label = label_encoder.inverse_transform([np.argmax(pred)])[0]
        print(f"\nText: {text}\nPredicted Label: {label} (Confidence: {np.max(pred)*100:.2f}%)")

# ------------------------------
# MAIN EXECUTION
# ------------------------------

texts, labels = load_data(DATA_PATH)
X, y, tokenizer, le, max_len = preprocess_texts(texts, labels)
vocab_size = len(tokenizer.word_index) + 1
num_classes = y.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42)

model = build_model(vocab_size, num_classes)

from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout 
# ADDED Dropout here
from sklearn.model_selection import train_test_split

# 1. Load dataset
# Make sure this path is correct or change to relative path "classification_dataset.csv"
# if model.py and the CSV are in the same directory and you run from that directory.
data = pd.read_csv("classification_dataset.csv") # specifiy dataset path under read_csv function 
texts = data['Text'].values
labels = data['Label'].values

# 2. Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_onehot = to_categorical(labels_encoded)
num_classes = labels_onehot.shape[1] # Store number of classes

# 3. Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
vocab_size = len(tokenizer.word_index) + 1 # Store vocab size

# 4. Pad sequences
max_len = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_len)

# --- ADD DIAGNOSTIC PRINTS HERE ---
print(f"Vocabulary size: {vocab_size}")
print(f"Max sequence length (max_len): {max_len}")
print(f"Number of classes: {num_classes}")
print(f"Shape of X (padded sequences): {X.shape}")
print(f"Shape of labels_onehot: {labels_onehot.shape}")
# --- END DIAGNOSTIC PRINTS ---

# 5. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, labels_onehot, test_size=0.2, random_state=42, stratify=labels_encoded) # Added stratify

# --- ADD MORE DIAGNOSTIC PRINTS FOR SPLIT DATA ---
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")
print("Training label distribution:")
print(pd.Series(np.argmax(y_train, axis=1)).value_counts(normalize=True).sort_index())
print("Test label distribution:")
print(pd.Series(np.argmax(y_test, axis=1)).value_counts(normalize=True).sort_index())
# --- END DIAGNOSTIC PRINTS FOR SPLIT DATA ---


# 6. Build RNN model
model = Sequential([
    # MODIFIED: Removed input_length
    Embedding(input_dim=vocab_size, output_dim=32), # Using vocab_size, reduced output_dim
    SimpleRNN(32), # Reduced RNN units
    Dropout(0.5),  # ADDED Dropout layer
    # MODIFIED: Using num_classes
    Dense(num_classes, activation='softmax')
])

# You can also explicitly build the model if needed, though fit() will do it.
# model.build(input_shape=(None, max_len)) # Optional explicit build

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=MAX_EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

model.save(MODEL_PATH)
save_label_map(le, LABEL_MAP_PATH)
plot_history(history)
demonstrate(model, tokenizer, le, max_len)
