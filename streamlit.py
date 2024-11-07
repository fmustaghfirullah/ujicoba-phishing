import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Konfigurasi dasar Streamlit
st.title("🛡️ Aplikasi Deteksi Phishing dengan Random Forest")
st.write("Aplikasi ini mendeteksi tautan phishing menggunakan model Random Forest Classifier.")

# Load dataset phishing.csv
@st.cache_data
def load_data():
    return pd.read_csv('phishing.csv')  # Pastikan file phishing.csv berada di direktori yang sesuai

data = load_data()

# Tampilkan sampel data
if st.checkbox("Tampilkan sampel data"):
    st.write(data.head())

# Pastikan kolom 'status' memiliki 1 untuk link aman dan 0 untuk phishing
# Pisahkan data menjadi fitur dan label
X = data['url']
y = data['status']

# Memuat model yang sudah dilatih
model = joblib.load('phishing_model.pkl')

# Memuat vectorizer untuk mentransformasikan URL
vectorizer = joblib.load('vectorizer.pkl')

# Text Vectorization menggunakan TF-IDF
X_transformed = vectorizer.transform(X)

# Evaluasi model untuk informasi umum
y_pred = model.predict(X_transformed)
accuracy = (y_pred == y).mean() * 100  # Akurasi model
st.subheader("📊 Hasil Evaluasi Model")
st.write(f"**Akurasi**: {accuracy:.2f}%")

# Menampilkan classification report (jika diperlukan)
from sklearn.metrics import classification_report
report = classification_report(y, y_pred)
st.text("Classification Report:")
st.text(report)

# Bagian Input URL untuk Deteksi Phishing
st.subheader("🔍 Deteksi Phishing dari URL")
input_url = st.text_input("Masukkan URL yang ingin diperiksa:")

if st.button("Deteksi"):
    if input_url:
        # Transformasi input URL
        input_transformed = vectorizer.transform([input_url])

        # Prediksi menggunakan model
        prediction = model.predict(input_transformed)[0]

        # Menampilkan hasil prediksi
        if prediction == 1:
            st.success("✅ URL ini terdeteksi sebagai **Aman**.")
        else:
            st.error("⚠️ URL ini terdeteksi sebagai **Phishing**.")
    else:
        st.write("⚠️ Silakan masukkan URL untuk pemeriksaan.")
