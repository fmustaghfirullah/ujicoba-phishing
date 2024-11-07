import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Preprocessing data
data['status'] = data['status'].map({'phishing': 0, 'legitimate': 1})  # Pastikan label sesuai dataset Anda

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
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Tampilkan hasil evaluasi model
st.subheader("📊 Hasil Evaluasi Model")
st.write(f"**Akurasi**: {accuracy * 100:.2f}%")
st.write(f"**Precision**: {precision:.2f}")
st.write(f"**Recall**: {recall:.2f}")
st.write(f"**F1-Score**: {f1:.2f}")

# Tampilkan classification report secara mendetail
st.subheader("📋 Classification Report")
st.json(report)

# Visualisasi salah satu pohon keputusan
st.subheader("🌳 Visualisasi Pohon Keputusan")
if st.checkbox("Tampilkan Visualisasi Salah Satu Pohon"):
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(model.estimators_[0], filled=True, max_depth=3, fontsize=10, feature_names=vectorizer.get_feature_names_out(), ax=ax)
    st.pyplot(fig)

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
        confidence = loaded_model.predict_proba(input_transformed)[0]

        # Tampilkan hasil prediksi dan confidence score
        if prediction == 1:
            st.error("⚠️ URL ini terdeteksi sebagai **Phishing**.")
        else:
            st.success("✅ URL ini terdeteksi sebagai **Legitimate**.")
        
        st.write(f"**Confidence Score**:")
        st.write(f"- Legitimate: {confidence[0]:.2f}")
        st.write(f"- Phishing: {confidence[1]:.2f}")
        
        # Tampilkan rincian skor prediksi
        st.write("### 🔎 Rincian Prediksi")
        st.write(f"- **Akurasi model**: {accuracy * 100:.2f}%")
        st.write(f"- **Precision**: {precision:.2f}")
        st.write(f"- **Recall**: {recall:.2f}")
        st.write(f"- **F1 Score**: {f1:.2f}")
        
        st.info("Catatan: Semakin tinggi skor confidence pada kategori 'Phishing', semakin besar kemungkinan URL tersebut berbahaya.")
    else:
        st.write("⚠️ Silakan masukkan URL untuk pemeriksaan.")
