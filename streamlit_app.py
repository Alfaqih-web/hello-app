import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


st.write(" 🦅🦅 Hello this is the App testing for the Diosease prediction")

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

# Data Preprocessing
def preprocess_data(data):
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train the model
@st.cache
def train_model(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Make a prediction
def make_prediction(model, user_input):
    prediction = model.predict([user_input])
    return prediction

# Streamlit App
st.title("Disease Classification App")

# Load and preprocess data
data = load_data()
X_train, X_test, y_train, y_test = preprocess_data(data)

# Train the model
model = train_model(X_train, y_train)

# Input from user
st.header("Enter Symptoms")
symptoms = []
for col in data.columns[:-1]:  # Assuming symptom columns are features
    value = st.number_input(f"{col} (0 or 1)", min_value=0, max_value=1, step=1)
    symptoms.append(value)

# Predict
if st.button("Predict Disease"):
    result = make_prediction(model, symptoms)
    st.write(f"Predicted Disease: {result[0]}")
