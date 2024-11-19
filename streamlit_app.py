import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

st.write(" 🦅🦅🦅 Hello Wthis is the App testing for the Diosease prediction")

st.info(" This app is to predict what type of disease you have through explain the symptoms experienced")

df = pd.read_csv('https://raw.githubusercontent.com/Alfaqih-web/hello-app/refs/heads/main/Symptom2Disease.csv')
df
