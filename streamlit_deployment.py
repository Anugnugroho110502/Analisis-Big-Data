import streamlit as st
import pandas as pd
import joblib
from googletrans import Translator
import matplotlib.pyplot as plt
import numpy as np

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
    
    # Model selection
    model_option = st.selectbox("Choose a model for prediction", ["Naive Bayes", "Logistic Regression"])

    if model_option == "Naive Bayes":
        prediction = nb_model.predict(user_review_tfidf)[0]
        probabilities = nb_model.predict_proba(user_review_tfidf)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"
    else:
        prediction = lr_model.predict(user_review_tfidf)[0]
        probabilities = lr_model.predict_proba(user_review_tfidf)[0]
        sentiment = "Positive" if prediction == 1 else "Negative"

    # Display the results
    st.write(f"{model_option} Prediction:", sentiment)

    # Display prediction probabilities
    st.write("Prediction Probabilities:")
    st.write(f"Positive: {probabilities[1]:.2f}, Negative: {probabilities[0]:.2f}")

    # Plotting probabilities
    st.write("Prediction Probabilities Bar Chart:")
    fig, ax = plt.subplots()
    categories = ['Negative', 'Positive']
    ax.bar(categories, probabilities, color=['red', 'green'])
    ax.set_ylim([0, 1])
    st.pyplot(fig)
    
    # Additional information
    st.write("Additional Information:")
    st.write(f"Length of the review: {len(user_review.split())} words")
    st.write(f"Model used: {model_option}")
