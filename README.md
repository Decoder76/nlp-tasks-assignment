Internship Assignment: Simple RNN for NLP Tasks

This repository contains the solution for the mandatory assignment: "Design and Implement a Simple (RNN)" which performs two distinct NLP tasks: Educational Text Classification and Next Word Generation.

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
