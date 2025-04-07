import streamlit as st
#TextBlob or Default
from textblob import TextBlob
# Naive bayes 
from textblob.classifiers import NaiveBayesClassifier
#BERT
from transformers import pipeline
#ollama 
import requests

#inports for bar chart or visualise page 
import pandas as pd
import plotly.express as px


# Sample training data for Naive Bayes
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

# Train the classifier
classifier = NaiveBayesClassifier(train_data)
#------------------------


st.title("Sentiment Analyzer App")
st.write("Enter any text below and find out its sentiment (positive, negative, or neutral).")
user_input = st.text_area("Enter your text here: ", height= 150)


#logic for bert
def load_pipeline():
    return pipeline("sentiment-analysis")

nlp = load_pipeline()

#logic for ollama

def ollama_sentiment_analysis(user_input):
    prompt = f"Classify the sentiment of the following text as positive, negative or neutral: {user_input}"
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

#@st.cache_resource
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


#logic after text box for backend 
option = st.radio("Choose the analysis method:", ("Default (TextBlob)", "Naive Bayes (trained)", "BERT", "Ollama"))
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please Enter some text first")
    else:
        if option == "Default (TextBlob)":
            blob = TextBlob(user_input)
            sentiment = blob.sentiment.polarity
            if sentiment > 0:
                st.success("Happy Happy Happyy")
            elif sentiment < 0:
                st.error("Sad Hu bhai")
            else:
                st.info("Neutral Librandu saade")
            st.write(f"sentiment: {sentiment:.3f}")
        elif option == "Naive Bayes (trained)":
            blob = TextBlob(user_input, classifier=classifier)
            result = blob.classify()
            if result == "pos":
                st.success("Happy Happy Happy")
            elif result == "neg":
                st.error("Sad Boi")
            else:
                st.info("Neutral Librandu")
            st.write(f"Polarity Score:  {result}")
        elif option == "BERT": 
            result_bert = nlp(user_input)[0]
            label = result_bert['label']
            score = result_bert['score']
            if label == "POSITIVE":
                st.success(f"🙂 Positive ({score:.2%} confidence)")
            elif label == "NEGATIVE":
                st.error(f"🙁 Negative ({score:.2%} confidence)")
            else:
                st.info(f"😐 Neutral-ish? ({score:.2%} confidence)")
            st.write("**Raw output:**", result_bert)
        else:
            st.info("Using LLM via Ollama...")
            result = ollama_sentiment_analysis(user_input)
            st.write("LLM Sentiment Prediction: ", result)

def show_sentiment_chart(textblob_result, naiveBayes_result, bert_result, ollama_result):
    models = ["Textblob", "Naive Bayes", "BERT", "Ollama"]
    sentiments = [textblob_result, naiveBayes_result, bert_result, ollama_result]
    if ollama_result:
        models.append("Ollama")
        sentiments.append(ollama_result)
    df = pd.DataFrame({
        "Model": models,
        "Sentiment": sentiments
    })
    fig = px.bar(
        df,
        x = "Model",
        color = "Sentiment",
        title = "Sentiment Prediction by each model",
        color_discrete_map={
            "pos": "green",
            "Positive": "green",
            "neg": "red",
            "Negative": "red",
            "neutral": "gray",
            "Neutral": "gray"
        }
    )
    st.plotly_chart(fig)
    

if st.button("Compare All Models"):
    if user_input.strip() == "":
        st.warning("Please Enter some text first!")
    else:
        #st.info("Default Textblob")
        tb = textblob_analysis(user_input)
        #st.info("Using Naive Bayes Classifier (Trained)")
        nb = naive_bayes_analysis(user_input)
        st.write(nb)
        #st.info("Using BERT")
        bt = bert_analysis(user_input)
        #st.info("Using LLM via Ollama...")
        ol = ollama_sentiment_analysis(user_input)
        #st.write("LLM Sentiment Prediction: ", result)
        cols = st.columns(4)
        cols[0].subheader("TextBlob")
        cols[0].write(tb)
        cols[1].subheader("Naive Bayes")
        cols[1].write(nb)
        cols[2].subheader("BERT")
        cols[2].write(bt)
        cols[3].subheader("Ollama (LLM)")
        cols[3].write(ol)
        show_sentiment_chart(tb,nb,bt,ol)
    

