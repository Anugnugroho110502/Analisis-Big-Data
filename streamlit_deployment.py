import streamlit as st
import pandas as pd
import joblib
from googletrans import Translator
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from wordcloud import WordCloud
from gtts import gTTS
import os

# Load the models and vectorizer
nb_model = joblib.load('sentiment_model_nb.pkl')
lr_model = joblib.load('sentiment_model_lr.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Initialize translator
translator = Translator()

# Function to translate review
def translate_review(text, dest_lang):
    translation = translator.translate(text, dest=dest_lang)
    return translation.text

# Function to convert text to speech
def text_to_speech(text, lang, filename):
    tts = gTTS(text=text, lang=lang)
    tts.save(filename)

# Streamlit app
st.title("IMDb Sentiment Analysis")
st.write("Analyze the sentiment of IMDb movie reviews.")

# Sidebar for model selection, language selection, and review input
st.sidebar.image("https://banner2.cleanpng.com/20180706/czk/kisspng-imdb-television-film-actor-imdb-5b3f4532486af5.0331136915308731382966.jpg", use_column_width=True)
st.sidebar.title("Settings")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")
model_option = st.sidebar.selectbox("Choose a model for prediction", ["Naive Bayes", "Logistic Regression"])
language_option = st.sidebar.selectbox("Choose a language for translation", ["Indonesian", "Spanish", "French", "German", "Japanese", "Javanese", "Korean"])
language_codes = {
    "Indonesian": "id",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Japanese": "ja",
    "Javanese": "jw",
    "Korean": "ko"
}
user_review = st.sidebar.text_area("Enter your review here:")

if user_review:
    # Translate review to selected language
    dest_lang = language_codes[language_option]
    translated_review = translate_review(user_review, dest_lang)
    st.write(f"Translated Review ({language_option}):\n\n", translated_review)

    # Convert translated review to speech
    audio_file = "translated_review.mp3"
    text_to_speech(translated_review, dest_lang, audio_file)
    audio_bytes = open(audio_file, 'rb').read()
    st.audio(audio_bytes, format='audio/mp3')

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

    # Generate and display WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(user_review)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    
    if sentiment == "Positive":
        plt.title('Word Cloud for Positive Review')
    else:
        plt.title('Word Cloud for Negative Review')
    
    st.pyplot(plt)

    # Additional information
    st.write("Additional Information:")
    st.write(f"Length of the review: {len(user_review.split())} words")
    st.write(f"Model used: {model_option}")

    # Remove the audio file after use
    os.remove(audio_file)
