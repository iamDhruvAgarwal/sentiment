import streamlit as st
from main import show_sentiment_chart
# Grab from session
# from main import tb,nb,bt,ol
tb = textblob_analysis(user_input)
nb = naive_bayes_analysis(user_input)
bt = bert_analysis(user_input)
ol = ollama_sentiment_analysis(user_input)
from components.analyzers import naive_bayes_analysis, textblob_analysis, bert_analysis, ollama_sentiment_analysis
if not all([tb,nb,bt]):
    st.warning("No sentiment results available. Please run analysis from the Main page first.")
    st.stop()

# then call your chart function
def show():
    show_sentiment_chart(tb,nb,bt,ol)
