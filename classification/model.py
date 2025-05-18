# model.py - RNN Text Classification (Math, Science, History)

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
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

# 7. Train the model
# Consider more epochs if you have more data, or fewer if it overfits too quickly
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test)) # Increased epochs for more learning opportunity

# 8. Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 9. Save results and label mapping
model.save("classification_rnn.keras") # CHANGED to .keras format as recommended
# For HDF5, use model.save("classification_rnn.h5")

with open("label_mapping.txt", "w") as f:
    for i, label in enumerate(le.classes_):
        f.write(f"{i}: {label}\n")
print("Saved model to classification_rnn.keras and label mapping to label_mapping.txt")


# 10. Demonstrate on new, unseen snippets
print("\n--- Demonstrating on new snippets ---")

new_snippets = [
    "The study of chemical reactions and compounds.",
    "Ancient Rome had a powerful army and built many roads.",
    "Solving for x in an equation like 2x + 5 = 11.",
    "Exploring the causes of the French Revolution.",
    "What is the force that keeps planets in orbit?", # Expected Science
    "The period of rebirth in European art and literature.", # Expected History
    "Calculating the area of a circle using pi." # Expected Math
]

# Preprocess new snippets
new_sequences = tokenizer.texts_to_sequences(new_snippets)
new_X = pad_sequences(new_sequences, maxlen=max_len)

# Predict
predictions = model.predict(new_X)
predicted_label_indices = np.argmax(predictions, axis=1)
predicted_labels = le.inverse_transform(predicted_label_indices)

for text, label, conf in zip(new_snippets, predicted_labels, predictions):
    print(f"Snippet: '{text}'")
    print(f"Predicted Label: {label} (Confidence: {np.max(conf)*100:.2f}%)")
    print(f"All confidences: {[(le.classes_[i], c*100) for i, c in enumerate(conf)]}\n")


print("Label Encoder Classes (for reference):")
for i, cls_name in enumerate(le.classes_):
    print(f"{i}: {cls_name}")

# Optional: Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")
print("Saved training history plot to training_history.png")
# plt.show() # Uncomment to display plot if running interactively
