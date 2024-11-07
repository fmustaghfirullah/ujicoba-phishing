# 🔗 Aplikasi Deteksi Link Phishing

![Streamlit](https://img.shields.io/badge/Streamlit-1.0-blue) ![RandomForest](https://img.shields.io/badge/Model-Random%20Forest-green) ![Python](https://img.shields.io/badge/Python-3.x-blue)

Aplikasi ini adalah alat deteksi tautan phishing berbasis web yang dikembangkan dengan Streamlit. Aplikasi ini menggunakan model **Random Forest** dan **analisis teks TF-IDF** untuk membantu pengguna mendeteksi tautan phishing dari input URL, memberikan hasil apakah tautan tersebut aman (legitimate) atau berbahaya (phishing). 💻🔒

## 🎯 Fitur
- **Prediksi phishing** berdasarkan analisis teks URL menggunakan model Random Forest.
- **Antarmuka pengguna** yang sederhana dan intuitif menggunakan Streamlit.
- **Akurasi prediksi** yang tinggi, hasil evaluasi diberikan secara langsung dalam aplikasi.
- **Visualisasi** sampel dataset, memungkinkan pengguna memahami data yang digunakan.

## 🚀 Cara Kerja Aplikasi
1. **Preprocessing Teks**: URL diubah menjadi representasi numerik menggunakan TF-IDF.
2. **Model Random Forest**: Model ini dilatih untuk memprediksi URL mana yang termasuk phishing atau legitimate.
3. **Prediksi Real-Time**: Pengguna memasukkan URL dan aplikasi langsung menampilkan hasil deteksi.

## 🛠️ Instalasi

1. Clone repository ini ke lokal Anda:
   ```bash
   git clone https://github.com/username/repo-name.git
   ```
2. Masuk ke folder project:
   ```bash
   cd repo-name
   ```
3. Install dependencies menggunakan `pip`:
   ```bash
   pip install -r requirements.txt
   ```
4. Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```

> **Note**: Pastikan dataset Anda tersedia dan disimpan dalam file `dataset.csv`. Struktur dataset diharapkan memiliki kolom `url` dan `label` (0 untuk legitimate, 1 untuk phishing).

## 📂 Struktur Project

```
├── app.py              # Script utama aplikasi Streamlit
├── dataset.csv         # Dataset untuk melatih model (tidak disertakan di repo, harap sediakan sendiri)
├── phishing_model.pkl  # Model yang telah dilatih (dihasilkan saat runtime)
├── vectorizer.pkl      # Vectorizer TF-IDF (dihasilkan saat runtime)
├── requirements.txt    # Daftar dependencies
└── README.md           # Dokumentasi project
```

## 📈 Evaluasi Model
Aplikasi akan menampilkan akurasi model secara otomatis setelah melatih model menggunakan data. Model ini dikembangkan dengan pendekatan sederhana, namun dapat diperluas untuk model yang lebih kompleks atau dataset yang lebih besar.

## 👤 Kontribusi
Kontribusi dalam bentuk ide, saran, atau peningkatan fitur sangat diharapkan! Silakan buka *issue* atau *pull request*.
