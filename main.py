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


#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



# Train the classifier //done in analyzers.py
#classifier = NaiveBayesClassifier(train_data)
#------------------------


#fetching user input from the text_area

st.title("Sentiment Analyzer App")
st.write("Enter any text below and find out its sentiment (positive, negative, or neutral).")
user_input = st.text_area("Enter your text here: ", height= 150)


#importing the MODELS

from components.analyzers import naive_bayes_analysis, textblob_analysis, bert_analysis, ollama_sentiment_analysis

    
#logic after text box for backend 
option = st.radio("Choose the analysis method:", ("Default (TextBlob)", "Naive Bayes (trained)", "BERT", "Ollama"))
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please Enter some text first")
    else:
        if option == "Default (TextBlob)":
            tb = textblob_analysis(user_input)
            st.write(tb)
        elif option == "Naive Bayes (trained)":
            nb = naive_bayes_analysis(user_input)
            st.write(nb)
        elif option == "BERT": 
            bt = bert_analysis(user_input)
            st.write(bt)
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
        #calling the chart function for visualise.py function
        show_sentiment_chart(tb,nb,bt,ol)



















# if st.button("Compare All Models"):
#     st.session_state["textblob_result"] = textblob_analysis(user_input)
#     st.session_state["naivBayes_result"] = naive_bayes_analysis(user_input)
#     st.session_state["bert_result"] = bert_analysis(user_input)
#     st.session_state["ollama_result"] = ollama_sentiment_analysis(user_input)
#     st.success("All Models Analyzed Now Redirecting to Visualise tab...")
#     st.write("Saved to session:", st.session_state)
# if st.button("Compare All Models"):
#     tb = textblob_analysis(user_input)
#     nb = naive_bayes_analysis(user_input)
#     bt = bert_analysis(user_input)
#     ol = ollama_sentiment_analysis(user_input)
#     st.success("All Models Analyzed Now Redirecting to Visualise tab...")
#     st.write("Saved to session:", st.session_state)
# page = st.sidebar.radio("Navigation",("Main","Visualise", "Export", "About"))
# from pages import visualize, export, about
# if page == "Main":
#     st.write("la")
# elif page == "Visualise":
#     visualize.show()