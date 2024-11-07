import streamlit as st
import joblib
import pandas as pd
import re

def extract_features(url):
    features = {}
    features['panjang_url'] = len(url)
    features['protokol'] = int("https" in url)
    features['sepsial_karakter'] = sum([1 for char in url if char in ['@', '-', '_', '%', '.', '=', '&']])
    features['jumlah_digit'] = sum([1 for char in url if char.isdigit()])
    features['jumlah_subdomain'] = url.count('.') - 1
    features['ipaddress'] = int(bool(re.search(r'[0-9]+(?:\.[0-9]+){3}', url)))
    return features

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
