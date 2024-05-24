import streamlit as st
import pandas as pd
import joblib
from googletrans import Translator

# Load the models and vectorizer
nb_model = joblib.load('sentiment_model_nb.pkl')
lr_model = joblib.load('sentiment_model_lr.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize translator
translator = Translator()

# Function to translate review to Indonesian
def translate_to_indonesian(text):
    translation = translator.translate(text, dest='id')
    return translation.text

# Streamlit app
st.title("IMDb Sentiment Analysis")
st.write("Analyze the sentiment of IMDb movie reviews.")

# Input text box for user review
user_review = st.text_area("Enter your review here:")

if user_review:
    # Translate review to Indonesian
    translated_review = translate_to_indonesian(user_review)
    st.write("Translated Review:", translated_review)

    # Vectorize the user input
    user_review_tfidf = vectorizer.transform([user_review])

    # Predict sentiment with Naive Bayes model
    nb_prediction = nb_model.predict(user_review_tfidf)[0]
    nb_sentiment = "Positive" if nb_prediction == 1 else "Negative"

    # Predict sentiment with Logistic Regression model
    lr_prediction = lr_model.predict(user_review_tfidf)[0]
    lr_sentiment = "Positive" if lr_prediction == 1 else "Negative"

    # Display the results
    st.write("Naive Bayes Prediction:", nb_sentiment)
    st.write("Logistic Regression Prediction:", lr_sentiment)
