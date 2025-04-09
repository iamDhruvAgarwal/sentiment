import streamlit as st
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from transformers import pipeline
import requests

def load_pipeline():
    return pipeline("sentiment-analysis")
nlp = load_pipeline()


def textblob_analysis(user_input):
    blob = TextBlob(user_input)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        result = "pos"
    elif sentiment < 0:
        result = "neg"
    else:
        result = "Neutral"
    #st.write(f"sentiment: {sentiment:.3f}")
    return result


def get_trained_classifier():
    # Replace this with your actual training data
    train_data = [
        ("I love this product", "pos"),
        ("This is an amazing experience", "pos"),
        ("I am so happy and excited", "pos"),
        ("This is bad", "neg"),
        ("I hate this", "neg"),
        ("This was a terrible experience", "neg"),
        ("I am happy", "pos"),
        ("I am not happy", "neg"),
        ("I don't like it", "neg"),
        ("It's okay", "neutral"),
        ("This is average", "neutral"),
        ("Nothing special", "neutral"),
    ]
    return NaiveBayesClassifier(train_data)

def naive_bayes_analysis(user_input):
    classifier = get_trained_classifier()
    blob = TextBlob(user_input, classifier=classifier)
    result = blob.classify()
    #st.write(f"Predicted Class: {result}")
    return result
    #return result.strip()


def bert_analysis(user_input):
    result_bert = nlp(user_input)[0]
    label = result_bert['label']
    score = result_bert['score']
    if label == "POSITIVE":
        result = (f"🙂 Positive ({score:.2%} confidence)")
    elif label == "NEGATIVE":
        result = (f"🙁 Negative ({score:.2%} confidence)")
    else:
        result = (f"😐 Neutral-ish? ({score:.2%} confidence)")
    #st.write("**Raw output:**", result_bert)
    return result


#logic for ollama

def ollama_sentiment_analysis(user_input):
    prompt = f"Classify the sentiment of the following text as positive, negative or neutral, answer in one word either positive, negative or neutral that's it: {user_input}"
    response = requests.post(
        "http://localhost:11434/api/generate",
        json = {
            "model": "mistral",
            "prompt": prompt,
            "stream": False
        }
    )
    result = response.json()["response"]
    return result.strip()
