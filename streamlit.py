import streamlit as st
import joblib
import pandas as pd
import re

def extract_features(url):
    # ... (your existing feature extraction function)


loaded_model = joblib.load('phishing_model.pkl')

st.title("Phishing URL Detection")
url = st.text_input("Enter a URL:")
if st.button("Predict"):
  if url:
    features = extract_features(url)
    input_df = pd.DataFrame([features])
    prediction = loaded_model.predict(input_df)
    if prediction[0] == 1:
        st.write("The URL is classified as phishing.")
    else:
        st.write("The URL is classified as safe.")

  else:
      st.write("Please enter a URL.")