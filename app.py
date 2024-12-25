import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the model and tokenizer
model = load_model("final_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def predict_next_words(input_text, num_words=10):
    """
    Predict the next `num_words` words based on the input text.
    """

    for _ in range(num_words):
        text_token = tokenizer.texts_to_sequences([input_text])[0]
        padded_text_token = pad_sequences([text_token], maxlen=18, padding='pre')
        prediction = model.predict(padded_text_token, verbose=0)
        pos = np.argmax(prediction)

        for word, idx in tokenizer.word_index.items():
            if idx == pos:
                input_text += " " + word

    return input_text

st.set_page_config(page_title="Next Word Prediction", layout="centered", page_icon="🔮")

# Header and Description
st.title("🔮 Next Word Prediction")
st.markdown(
    """
    Welcome to the **Next Word Prediction App**! 🚀  
    Enter a sequence of words, and the AI model will predict the next possible words for you.
    """
)

# Input text box
input_text = st.text_input("💡Enter your text:", placeholder="eg. The weather is nice today.")

# Number of words to predict
num_words = st.slider("Number of words to predict:", 1, 20, 10)

if st.button("Predict"):
    if input_text.strip():
        with st.spinner("Predicting..."):
            result = predict_next_words(input_text, num_words)
        st.success("✨ Prediction Complete!")
        st.markdown(f"**Predicted Text : ** {result}")
    else:
        st.error("Please enter some text to start predicting.")

st.markdown(
    """
    ---
    Developed with ❤️ using **Streamlit**  
    """
)
