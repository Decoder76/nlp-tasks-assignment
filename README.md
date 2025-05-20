
# 🧠 NLP with RNN – Text Classification & Generation
=======
Internship Assignment: Simple RNN for NLP Tasks


### Internship Assignment – Sutram Solutions Pvt. Ltd.  
**Submitted by:** Lokesh
**Track:** Applied AI/ML – NLP Focus  


---

## 📌 Overview

This project demonstrates the use of **Recurrent Neural Networks (RNNs)** and **LSTM** in solving two fundamental Natural Language Processing (NLP) tasks:

1. **Text Classification**  
   Classifies educational text snippets into one of three subjects: **Math, Science, History**.

2. **Next Word Prediction**  
   Given an initial seed phrase, the model generates the next **20 words**, simulating intelligent content prediction.

The models are implemented using **Keras** with TensorFlow backend, trained on synthetically curated educational data.

---

## 📁 Project Structure

```
rnn_assignment/
│
├── classification/                    # Text classification module
│   ├── classification_dataset.csv     # Input dataset
│   ├── model.py                       # Full training + prediction script
│   ├── classification_rnn.keras       # Saved model
│   ├── classification_rnn.h5          # Saved model (alternate format)
│   ├── label_mapping.txt              # Encoded labels
│   └── training_history.png           # Training graph

├── generation/                        # Text generation module
│   ├── science_corpus.txt             # Training corpus (science-based)
│   ├── model.py                       # Full training + text generation
│   ├── generation_rnn_model.keras     # Trained model
│   ├── generation_tokenizer.pkl       # Tokenizer for inference

├── demo.gif / demo_rnn_assignment_sample.mp4  # Demo media
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
```

---

## 🧩 Task 1: Educational Text Classification

### 🔍 Objective:
Train a model to predict whether a sentence is about **Math**, **Science**, or **History**.

### 🔧 How It Works:
- Input data is tokenized and padded.
- Labels are one-hot encoded using `LabelEncoder`.
- A **Simple RNN** with embedding and dropout layers is trained on the dataset.
- The model evaluates unseen snippets and predicts the subject with probability.

### 📈 Sample Result:
```
Input: "Solving for x in an equation like 2x + 5 = 11"
Predicted Label: Math (Confidence: 98.45%)
```

---

## 🧩 Task 2: Next Word Generation

### 🔍 Objective:
Build a model that, given a seed sentence, generates 20 coherent words to simulate educational content continuation.

### 🔧 How It Works:
- A scientific corpus is preprocessed and tokenized.
- Word sequences of length 10 are used as training input.
- A **2-layer LSTM** network predicts the next word.
- At inference, the model generates a sentence based on the given input using temperature-based sampling.

### 📈 Sample Result:
```
Seed: "Photosynthesis is the process by which"
Generated: "plants absorb sunlight and convert it into food through chlorophyll during chemical reactions in green leaves"
```

---

## 🚀 How to Run

### 🔨 Install Dependencies
```bash
pip install -r requirements.txt
```

### ▶️ Train & Test Classification Model
```bash
cd classification
python model.py
```

### ▶️ Train & Generate Text
```bash
cd generation
python model.py
```

---

## 📌 Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- NumPy
- scikit-learn
- Matplotlib

---

## 📊 Notes

- All data used is synthetic and educational.
- The models are kept simple for clarity but can be extended using GRUs, BERT, or Transformer-based architectures.
- Includes diagnostics, plots, and confidence scores for better interpretability.

---

## 🔗 Useful Links

- [Classification Branch](https://github.com/Decoder76/nlp-tasks-assignment/tree/lokesh_jayswal_nlptasks/classification)
- [Generation Branch](https://github.com/Decoder76/nlp-tasks-assignment/tree/lokesh_jayswal_nlptasks/generation)
- [Full Repo](https://github.com/Decoder76/nlp-tasks-assignment/tree/lokesh_jayswal_nlptasks)

---

## ✅ Deliverables

- [x] Functional RNN model for text classification  
- [x] Functional LSTM model for next-word prediction  
- [x] Model evaluation, visualization, and inference examples  
- [x] Documentation and source code  
- [x] GitHub submission and PDF demo

---

> “Focus on fundamentals and clarity — that's where real AI starts.”  
=======
## Setup and Dependencies

This project uses Python and several common data science/machine learning libraries.

1.  **Clone the repository (or ensure you have the files).**
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `tensorflow` includes `keras`)*

## Task 1: Educational Text Classification

This task involves classifying short educational text snippets into one of three predefined categories: Math, Science, or History.

**Model Architecture:**
*   A Sequential Keras model is used.
*   **Embedding Layer:** Converts word indices to dense vectors (e.g., 32 dimensions).
*   **SimpleRNN Layer:** A simple Recurrent Neural Network layer (e.g., 32 units) to process the sequence.
*   **Dropout Layer:** Added to help prevent overfitting.
*   **Dense Output Layer:** A `softmax` activated layer with 3 units, corresponding to the number of classes.

**Dataset:**
*   The `classification/classification_dataset.csv` file contains text snippets and their corresponding labels.
*   *Current dataset size: [Specify number of rows, e.g., 30 samples (10 per class). You might want to update this if you expanded it.]*

**Preprocessing Steps:**
1.  Labels are encoded using `sklearn.preprocessing.LabelEncoder` and then one-hot encoded.
2.  Text is tokenized using `tensorflow.keras.preprocessing.text.Tokenizer`.
3.  Sequences are padded to a uniform length (`max_len`) using `pad_sequences`.
4.  Data is split into training and testing sets.

**How to Run:**
Navigate to the `rnn_assignment/` directory in your terminal.
```bash
python classification/model.py

