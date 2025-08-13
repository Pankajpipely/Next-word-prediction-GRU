import streamlit as st
import numpy as np
import pickle
import sys
import tensorflow.keras.preprocessing as keras_preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU
from tensorflow.keras.preprocessing.sequence import pad_sequences

# --- Patch for old pickled tokenizer paths ---
sys.modules['keras.src.preprocessing'] = keras_preprocessing

# Custom GRU wrapper to ignore unsupported 'time_major' argument
class GRUCompatible(GRU):
    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)  # Remove unsupported arg
        super().__init__(*args, **kwargs)

# --- Load model safely without triggering rebuild issues ---
try:
    model = load_model(
        "next_word_GRU.h5",
        custom_objects={"GRU": GRUCompatible},
        compile=False  # prevents rebuild bug
    )
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Load tokenizer ---
try:
    with open("tokenizer.pickle", "rb") as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    st.stop()

# --- Function to predict next word ---
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if not token_list:
        return None  # No tokens found

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len - 1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding="pre")
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = int(np.argmax(predicted, axis=1)[0])

    # Reverse lookup in tokenizer
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# --- Streamlit app ---
st.title("Next Word Prediction With GRU")
input_text = st.text_input("Enter the sequence of words", "To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    if next_word:
        st.success(f"Next Word: {next_word}")
    else:
        st.warning("Could not predict the next word.")

