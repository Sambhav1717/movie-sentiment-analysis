import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stop_words]
    return " ".join(text)

st.title("🎬 Movie Sentiment Analysis App")

user_input = st.text_area("Enter a sentence:")

if st.button("Predict Sentiment"):
    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    sentiment_map = {
        0: "Very Negative 😡",
        1: "Negative 😕",
        2: "Neutral 😐",
        3: "Positive 🙂",
        4: "Very Positive 😄"
    }

    st.success(f"Prediction: {sentiment_map[prediction]}")