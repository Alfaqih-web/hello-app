import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from wordcloud import WordCloud



st.write(" 🦅🦅🦅 Hello Wthis is the App testing for the Diosease prediction")

st.info(" This app is to predict what type of disease you have through explain the symptoms experienced")

with st.expander("Data"):
  st.write("**Raw Data**")
  df = pd.read_csv('https://raw.githubusercontent.com/Alfaqih-web/hello-app/refs/heads/main/Symptom2Disease.csv')
  df
  
  st.write("**X**")
  X = df.drop("label", axis=1)
  X
  
  st.write("**Y**")
  y = df.label
  y
