import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Konfigurasi dasar Streamlit
st.title("🛡️ Aplikasi Deteksi Phishing dengan Random Forest")
st.write("Aplikasi ini mendeteksi tautan phishing menggunakan model Random Forest Classifier.")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv('phishing.csv')  # Pastikan dataset Anda ada di lokasi yang sesuai

data = load_data()

# Tampilkan sampel data
if st.checkbox("Tampilkan sampel data"):
    st.write(data.head())

# Preprocessing data: pastikan tidak ada NaN pada kolom 'label'
data['status'] = data['status'].map({'phishing': 0, 'legitimate': 1})  # Pastikan label sesuai dataset Anda

# Hapus baris dengan NaN pada 'label' dan fitur lainnya
data = data.dropna(subset=['label', 'status'])

# Pisahkan data menjadi fitur dan label
X = data['url']
y = data['status']

# Text Vectorization menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(X)

# Split data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Train model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model dan vectorizer
joblib.dump(model, 'phishing_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Evaluasi model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Tampilkan hasil evaluasi model
st.subheader("📊 Hasil Evaluasi Model")
st.write(f"**Akurasi**: {accuracy * 100:.2f}%")
st.text("Classification Report:")
st.text(report)

# Bagian Input URL untuk Deteksi
st.subheader("🔍 Deteksi Phishing dari URL")
input_url = st.text_input("Masukkan URL yang ingin diperiksa:")

if st.button("Deteksi"):
    if input_url:
        # Muat model dan vectorizer yang sudah disimpan
        loaded_model = joblib.load('phishing_model.pkl')
        loaded_vectorizer = joblib.load('vectorizer.pkl')
        
        # Transformasi input URL
        input_transformed = loaded_vectorizer.transform([input_url])
        
        # Prediksi
        prediction = loaded_model.predict(input_transformed)[0]
        
        if prediction == 1:
            st.error("⚠️ URL ini terdeteksi sebagai **Phishing**.")
        else:
            st.success("✅ URL ini terdeteksi sebagai **Legitimate**.")
    else:
        st.write("⚠️ Silakan masukkan URL untuk pemeriksaan.")
