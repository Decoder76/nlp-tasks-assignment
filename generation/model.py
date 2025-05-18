import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, Dropout, LSTM # LSTM can be better for generation
import re

# --- 1. Load and Preprocess Text Data ---
def load_and_preprocess_text(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Basic cleaning: lowercasing and removing special characters (optional, can be refined)
    text = text.lower()
    text = re.sub(r'[^a-z\s\.]', '', text) # Keep letters, spaces, and periods
    text = re.sub(r'\s+', ' ', text).strip() # Remove multiple spaces
    return text

corpus_filepath = "generation/science_corpus.txt" # Make sure this path is correct
text_data = load_and_preprocess_text(corpus_filepath)

if not text_data:
    print(f"Error: Corpus file '{corpus_filepath}' is empty or not found.")
    exit()

# --- 2. Tokenize the Corpus ---
tokenizer = Tokenizer(oov_token="<unk>") # oov_token handles unknown words during generation
tokenizer.fit_on_texts([text_data]) # Pass as a list
total_words = len(tokenizer.word_index) + 1 # +1 for the 0 index (padding or <unk> if not using oov_token explicitly)

print(f"Total unique words (vocab size): {total_words}")

# --- 3. Create Input Sequences and Target Words ---
input_sequences = []
# Split the text into sentences or use a sliding window over tokens
# Using a simple token-based sliding window for now
tokens = tokenizer.texts_to_sequences([text_data])[0]

seq_length = 10  # Length of input sequence (e.g., predict 11th word from 10 words)
# You can experiment with different sequence lengths

if len(tokens) <= seq_length:
    print(f"Error: The corpus has {len(tokens)} tokens, which is not enough for a sequence length of {seq_length + 1}.")
    print("Please provide a larger corpus.")
    exit()

for i in range(seq_length, len(tokens)):
    # Extract a sequence of 'seq_length' words
    sequence = tokens[i-seq_length:i]
    # The next word is the target
    target_word_idx = tokens[i]
    input_sequences.append(sequence)
    # We will one-hot encode the target word later or use sparse categorical crossentropy

X = np.array(input_sequences)
# Prepare targets (y) - we'll one-hot encode them
y = np.array([tokens[i] for i in range(seq_length, len(tokens))])
y_categorical = to_categorical(y, num_classes=total_words)


print(f"Number of input sequences: {len(X)}")
if len(X) == 0:
    print("No sequences generated. Check corpus size and seq_length.")
    exit()

print(f"Shape of X: {X.shape}") # (num_sequences, seq_length)
print(f"Shape of y_categorical: {y_categorical.shape}") # (num_sequences, total_words)

# --- 4. Build RNN/LSTM Model ---
# You can try SimpleRNN first, but LSTM/GRU are generally better for text generation
embedding_dim = 50 # Or 64, 100 etc.
rnn_units = 128   # Or 64, 256 etc.

model = Sequential([
    Embedding(input_dim=total_words, output_dim=embedding_dim),    # SimpleRNN(rnn_units),
    LSTM(rnn_units, return_sequences=True), # if stacking LSTMs
    Dropout(0.2),
    LSTM(rnn_units), # Using LSTM
    Dropout(0.2),
    Dense(total_words, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 5. Train the Model ---
epochs = 50  # Adjust based on dataset size and observed performance
batch_size = 32 # Adjust based on dataset size and memory

# NOTE: Training can be time-consuming for text generation, especially with larger corpora/models.
# For a very small corpus like the example, it might overfit quickly or not learn much.
print("\nStarting model training...")
try:
    history = model.fit(X, y_categorical, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save("generation_rnn_model.keras")
    print("Model training complete and saved as generation_rnn_model.keras.")

    # Save the tokenizer
    import pickle
    with open('generation_tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Tokenizer saved as generation_tokenizer.pkl.")

except Exception as e:
    print(f"An error occurred during training: {e}")
    exit()


# --- 6. Implement a Function to Generate Text ---
def generate_text(seed_text, next_words, model, tokenizer, max_sequence_len):
    print(f"\n--- Generating text from seed: '{seed_text}' ---")
    generated_text = seed_text.lower() # Start with lowercase seed
    
    for _ in range(next_words):
        # Tokenize the current sequence
        token_list = tokenizer.texts_to_sequences([generated_text])[0]
        
        # Pad the sequence
        # We need to ensure it's the same length as the training sequences
        # If the current sequence is longer than max_sequence_len, truncate it from the beginning
        if len(token_list) > max_sequence_len:
            token_list = token_list[-max_sequence_len:]
        
        # Pad if shorter
        token_list_padded = pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        
        # Predict the next word
        predicted_probabilities = model.predict(token_list_padded, verbose=0)[0]
        
        # Inside generate_text, after getting probabilities
        temperature = 0.5 # Experiment with values like 0.2, 0.5, 1.0, 1.2
        scaled_probabilities = np.log(predicted_probabilities) / temperature
        exp_probabilities = np.exp(scaled_probabilities)
        final_probabilities = exp_probabilities / np.sum(exp_probabilities)
        predicted_index = np.random.choice(len(final_probabilities), p=final_probabilities)
        # Sample a word from the predicted probabilities (could also use np.argmax for deterministic)
        # predicted_index = np.argmax(predicted_probabilities)
        
        # For more diverse (but potentially less coherent) text, sample based on probabilities
        # Avoid 0 index if it's padding or <unk> is not desired as output start
        # Filter out <unk> token if you don't want it in generation
        if "<unk>" in tokenizer.word_index:
            predicted_probabilities[tokenizer.word_index["<unk>"]] = 0
            predicted_probabilities /= np.sum(predicted_probabilities) # Re-normalize

       #predicted_index = np.random.choice(len(predicted_probabilities), p=predicted_probabilities)

        # Convert index to word
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_index:
                output_word = word
                break
        
        if output_word:
            generated_text += " " + output_word
        else:
            # If for some reason the word isn't found (shouldn't happen if index is valid)
            break 
            
    return generated_text.strip()

# --- 7. Demonstrate Text Generation ---
# Ensure the model is trained before calling generate_text
# Pick a seed text that exists in your vocabulary or is common
# The seed text should ideally be close to seq_length words
seed_example = " ".join(text_data.split()[:seq_length]) if len(text_data.split()) >= seq_length else "physics explores the"

if hasattr(model, 'layers'): # Check if model is trained/loaded
    generated_output = generate_text(seed_example, 20, model, tokenizer, seq_length)
    print("\nGenerated Output:")
    print(generated_output)

    print("\nTo generate text again later with saved model and tokenizer:")
    print("1. Load the model: `tf.keras.models.load_model('generation_rnn_model.keras')`")
    print("2. Load the tokenizer: `with open('generation_tokenizer.pkl', 'rb') as handle: tokenizer = pickle.load(handle)`")
    print("3. Call `generate_text(...)` function.")
else:
    print("Model not trained. Skipping generation demonstration.")
