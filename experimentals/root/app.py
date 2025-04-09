import streamlit as st
from .pages import main
from pages import visualize, export, about  # assuming these are functions

#st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main", "Visualize", "Export", "About"])

if page == "Main":
    main.show()
elif page == "Visualize":
    visualize.show()
elif page == "Export":
    export.show()
elif page == "About":
    about.show()
