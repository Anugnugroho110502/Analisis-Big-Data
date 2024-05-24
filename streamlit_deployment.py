import streamlit as st
import pandas as pd
import joblib
from googletrans import Translator
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud

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

# Sidebar for model selection, wordcloud settings, and review input
st.sidebar.title("Settings")
user_review = st.sidebar.text_area("Enter your review here:")
model_option = st.sidebar.selectbox("Choose a model for prediction", ["Naive Bayes", "Logistic Regression"])
min_word_freq = st.sidebar.slider("Minimum word frequency for WordCloud", 1, 10, 1)

if user_review:
    # Translate review to Indonesian
    translated_review = translate_to_indonesian(user_review)
    st.write("Translated Review:", translated_review)

    # Vectorize the user input
    user_review_tfidf = vectorizer.transform([user_review])

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

    # Display prediction probabilities using Plotly
    st.write("Prediction Probabilities:")
    fig = go.Figure(data=[
        go.Bar(name='Negative', x=['Negative'], y=[probabilities[0]], marker_color='red'),
        go.Bar(name='Positive', x=['Positive'], y=[probabilities[1]], marker_color='green')
    ])
    fig.update_layout(barmode='group', xaxis_title='Sentiment', yaxis_title='Probability', yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig)

    # Generate and display WordCloud for positive reviews
    if sentiment == "Positive":
        wordcloud = WordCloud(width=800, height=400, background_color='white', min_word_length=min_word_freq).generate(user_review)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud for Positive Review')
        st.pyplot(plt)

    # Additional information
    st.write("Additional Information:")
    st.write(f"Length of the review: {len(user_review.split())} words")
    st.write(f"Model used: {model_option}")
