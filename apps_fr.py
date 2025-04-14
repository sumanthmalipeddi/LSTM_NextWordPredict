import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM Model
model = load_model('next_word_lstm.h5')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# --- Custom Styling ---
st.markdown("""
    <style>
    body {
        background-color: #f5f5f5;
    }
    .main {
        background-color: #f5f5f5;
        padding: 20px;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        color: #004d40;
        margin-bottom: 20px;
    }
    .section {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 25px;
    }
    .info {
        font-size: 16px;
        color: #333333;
    }
    .output {
        font-size: 22px;
        font-weight: 600;
        color: #795548;
        background-color: #fffde7;
        padding: 12px 20px;
        border-radius: 10px;
        display: inline-block;
        margin-top: 15px;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #333333;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 0.5rem;
    }
    .stButton > button {
        background-color: #004d40;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #00695c;
        transition: 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

# --- App Layout ---
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.markdown("<div class='title'>Next Word Prediction with LSTM</div>", unsafe_allow_html=True)

# --- Model Info Section ---
st.markdown("""
<div class='section info'>
    <h4>🔍 About the Model</h4>
    <p>
        This application uses a <b>Long Short-Term Memory (LSTM)</b> deep learning model — a type of recurrent neural network (RNN) 
        that excels in understanding and generating text sequences. LSTM models are especially good at learning long-range 
        dependencies in text, making them ideal for language-based tasks like next-word prediction.
    </p>
</div>
""", unsafe_allow_html=True)

# --- Input Section ---

# Custom label in black
st.markdown("<span style='font-weight:600; font-size:16px; color:#000000;'>🖊️ Enter a sequence of words</span>", unsafe_allow_html=True)

# Streamlit input (with label turned off by giving it an empty string)
input_text = st.text_input("", "To be or not to")

if st.button("🔮 Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)

    if next_word:
        st.markdown(f"<div class='output'>✨ Predicted Next Word: {next_word}</div>", unsafe_allow_html=True)
    else:
        st.error("Could not predict the next word. Please try a different phrase.")

st.markdown("</div>", unsafe_allow_html=True)