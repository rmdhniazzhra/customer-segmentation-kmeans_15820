# Mall Customer Segmentation using K-Means Clustering

Aplikasi berbasis web untuk mengelompokkan pelanggan mall berdasarkan karakteristik *Annual Income* dan *Spending Score* menggunakan algoritma K-Means.

## 🔗 Live Demo
Aplikasi web yang telah di-deploy dapat diakses melalui tautan berikut:  
👉 **[Mall Customer Segmentation Web App](https://customer-segmentation-15820.streamlit.app/)**

## 📂 Struktur Repositori
* `data/` : Berisi dataset mentah `Mall_Customers.csv`.
* `notebooks/` : Berisi notebook analisis eksplorasi data (EDA) dan pengujian Elbow Method.
* `model_kmeans.pkl` : Model hasil pelatihan K-Means yang disimpan dalam format biner pickle.
* `app.py` : Script antarmuka aplikasi web berbasis Streamlit.
* `requirements.txt` : Daftar pustaka dependensi Python untuk deployment.

## ⚙️ Cara Menjalankan Secara Lokal

Jalankan perintah berikut di terminal:

```bash
# 1. Clone repositori ini
git clone [https://github.com/rmdhniazzhra/customer-segmentation-kmeans_15820.git](https://github.com/rmdhniazzhra/customer-segmentation-kmeans_15820.git)

# 2. Masuk ke direktori proyek
cd customer-segmentation-kmeans_15820

# 3. Pasang pustaka dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi Streamlit
streamlit run app.py
