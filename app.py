import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os

st.set_page_config(
    page_title="Mall Customer Intelligence Platform",
    page_icon="🛍️",
    layout="wide"
)

# ---------------------------------------------------------
# Load Dataset & Model
# ---------------------------------------------------------
@st.cache_data
def load_data():
    dataset_path = os.path.join("data", "Mall_Customers.csv")
    if os.path.exists(dataset_path):
        return pd.read_csv(dataset_path)
    return pd.read_csv("Mall_Customers.csv")

@st.cache_resource
def load_model():
    with open("model_kmeans.pkl", "rb") as file:
        return pickle.load(file)

try:
    df = load_data()
    kmeans = load_model()
except Exception as e:
    st.error(f"Gagal memuat file dataset atau model: {e}")
    st.stop()

# ---------------------------------------------------------
# Label & Karakteristik Klaster
# ---------------------------------------------------------
cluster_info = {
    0: {
        "nama": "Cluster 1: Standar / Rata-rata",
        "deskripsi": "Pelanggan dengan tingkat pendapatan tahunan dan skor pengeluaran yang berada pada rentang menengah.",
        "strategi": "Pertahankan loyalitas dengan promo berkala, program reward poin belanja, dan diskon musiman untuk menjaga kestabilan transaksi."
    },
    1: {
        "nama": "Cluster 2: Prioritas / Target Utama (Sultan)",
        "deskripsi": "Pelanggan berpendapatan tinggi dan memiliki intensitas pengeluaran belanja yang sangat tinggi.",
        "strategi": "Berikan layanan VIP / concierge, undangan peluncuran produk edisi terbatas, program loyalitas eksklusif, dan penawaran kategori luxury."
    },
    2: {
        "nama": "Cluster 3: Boros / Impulsif",
        "deskripsi": "Pelanggan dengan pendapatan relatif rendah, namun memiliki kecenderungan pengeluaran belanja yang sangat tinggi.",
        "strategi": "Tawarkan produk tren terkini, flash sale akhir pekan, bundling paket hemat, serta opsi pembayaran digital/paylater dengan bunga rendah."
    },
    3: {
        "nama": "Cluster 4: Berhati-hati / Finansial Sadar",
        "deskripsi": "Pelanggan dengan pendapatan tahunan tinggi, tetapi memiliki pengeluaran belanja yang sangat hemat/rendah.",
        "strategi": "Tawarkan produk bernilai investasi jangka panjang, program diskon bernilai tambah tinggi, dan promosi yang menonjolkan efisiensi fungsional barang."
    },
    4: {
        "nama": "Cluster 5: Hemat / Rasional Rendah",
        "deskripsi": "Pelanggan dengan pendapatan tahunan rendah serta skor pengeluaran belanja yang juga rendah.",
        "strategi": "Fokus pada produk kebutuhan pokok harian (*fast-moving consumer goods*), penawaran harga termurah, dan voucher potongan langsung."
    }
}

# ---------------------------------------------------------
# Navigasi Sidebar (Multi-Fitur)
# ---------------------------------------------------------
st.sidebar.title("Navigasi Sistem")
menu = st.sidebar.radio(
    "Pilih Modul Fitur:",
    [
        "Fitur 1: Prediksi & Segmentasi Interaktif",
        "Fitur 2: Analisis Data & Elbow Curve",
        "Fitur 3: Rekomendasi Strategi Pemasaran"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption("Sistem Segmentasi Pelanggan Mall Berbasis K-Means Clustering")

# ---------------------------------------------------------
# FITUR 1: PREDIKSI & SEGMENTASI INTERAKTIF
# ---------------------------------------------------------
if menu == "Fitur 1: Prediksi & Segmentasi Interaktif":
    st.title("🛍️ Segmentasi Pelanggan Mall (K-Means)")
    st.markdown("Aplikasi web untuk mengelompokkan pelanggan mall berdasarkan karakteristik **Annual Income** dan **Spending Score** secara interaktif.")
    
    col_input, col_result = st.columns([1, 2])
    
    with col_input:
        st.subheader("Input Data Pelanggan Baru")
        income = st.slider("Annual Income (k$)", min_value=10, max_value=150, value=60, step=1)
        spending = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=50, step=1)
        btn_predict = st.button("Prediksi Klaster", use_container_width=True)
    
    with col_result:
        input_data = np.array([[income, spending]])
        pred = kmeans.predict(input_data)[0]
        info = cluster_info.get(pred, cluster_info[0])
        
        st.subheader("Hasil Segmentasi")
        st.success(f"**{info['nama']}**")
        st.markdown(f"**Karakteristik:** {info['deskripsi']}")
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Annual Income", f"${income}k")
        col_m2.metric("Spending Score", f"{spending}")
        
        st.subheader("Posisi Pelanggan pada Sebaran Data")
        
        # Buat visualisasi sebaran klaster
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Fitur untuk pemetaan data
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
        labels = kmeans.labels_
        colors = ['#e74c3c', '#2980b9', '#27ae60', '#16a085', '#8e44ad']
        
        for k in range(5):
            ax.scatter(
                X[labels == k, 0], 
                X[labels == k, 1], 
                s=35, 
                c=colors[k], 
                label=f'Cluster {k+1}', 
                alpha=0.75
            )
            
        centroids = kmeans.cluster_centers_
        ax.scatter(
            centroids[:, 0], 
            centroids[:, 1], 
            s=180, 
            c='gold', 
            marker='X', 
            edgecolor='black', 
            label='Centroids'
        )
        
        ax.scatter(
            income, 
            spending, 
            s=220, 
            c='black', 
            marker='*', 
            label='Input Pelanggan', 
            edgecolor='yellow'
        )
        
        ax.set_title("Visualisasi Klasterisasi K-Means", fontsize=12)
        ax.set_xlabel("Annual Income (k$)", fontsize=10)
        ax.set_ylabel("Spending Score (1-100)", fontsize=10)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.5)
        
        st.pyplot(fig)

# ---------------------------------------------------------
# FITUR 2: ANALISIS DATA & ELBOW CURVE
# ---------------------------------------------------------
elif menu == "Fitur 2: Analisis Data & Elbow Curve":
    st.title("📊 Eksplorasi Data & Penentuan Klaster Optimal")
    st.markdown("Modul analitik untuk mengevaluasi distribusi dataset pelanggan mall dan memverifikasi jumlah klaster optimal menggunakan **Elbow Method**.")
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    col_stat1.metric("Total Data Pelanggan", f"{len(df)} Baris")
    col_stat2.metric("Rata-rata Pendapatan", f"${df['Annual Income (k$)'].mean():.1f}k")
    col_stat3.metric("Rata-rata Spending Score", f"{df['Spending Score (1-100)'].mean():.1f}")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("Dataset Mall Customers")
        st.dataframe(df, use_container_width=True, height=350)
        
    with col_right:
        st.subheader("Grafik Elbow Method (Evaluasi Nilai K)")
        
        # Hitung WCSS untuk K=1 s/d 10
        X = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
        wcss = []
        for i in range(1, 11):
            from sklearn.cluster import KMeans
            km_temp = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
            km_temp.fit(X)
            wcss.append(km_temp.inertia_)
            
        fig_elbow, ax_elbow = plt.subplots(figsize=(6, 4.4))
        ax_elbow.plot(range(1, 11), wcss, marker='o', color='#2980b9', linewidth=2)
        ax_elbow.axvline(x=5, color='#e74c3c', linestyle='--', label='K Optimal = 5')
        ax_elbow.set_title("Metode Elbow untuk K Optimal", fontsize=11)
        ax_elbow.set_xlabel("Jumlah Klaster (K)", fontsize=9)
        ax_elbow.set_ylabel("WCSS (Within-Cluster Sum of Square)", fontsize=9)
        ax_elbow.grid(True, linestyle="--", alpha=0.5)
        ax_elbow.legend()
        st.pyplot(fig_elbow)
        
    st.info("💡 **Dasar Ilmiah Pemilihan K=5:** Berdasarkan kurva di atas, penurunan inersia (WCSS) mulai melandai secara signifikan setelah titik k = 5 (membentuk sudut siku 'elbow'), sehingga pembagian menjadi 5 segmen dinilai paling ideal dan seimbang.")

# ---------------------------------------------------------
# FITUR 3: REKOMENDASI STRATEGI PEMASARAN
# ---------------------------------------------------------
elif menu == "Fitur 3: Rekomendasi Strategi Pemasaran":
    st.title("💡 Rekomendasi Strategi Bisnis & Pemasaran")
    st.markdown("Modul pengambil keputusan manajerial untuk memberikan rekomendasi aksi promosi terarah berdasarkan segmen pelanggan mall.")
    
    selected_cluster = st.selectbox(
        "Pilih Segmen Klaster Pelanggan:",
        options=list(cluster_info.keys()),
        format_func=lambda x: cluster_info[x]["nama"]
    )
    
    info_c = cluster_info[selected_cluster]
    
    col_d1, col_d2 = st.columns([1, 1])
    
    with col_d1:
        st.subheader("📋 Profil Segmen")
        st.write(info_c["deskripsi"])
        
        # Filter data sesuai klaster
        X_all = df[['Annual Income (k$)', 'Spending Score (1-100)']].values
        df_display = df.copy()
        df_display['Cluster'] = kmeans.labels_ + 1
        subset = df_display[df_display['Cluster'] == (selected_cluster + 1)]
        
        st.metric("Jumlah Pelanggan di Klaster Ini", f"{len(subset)} Orang")
        st.dataframe(subset[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].head(6), use_container_width=True)
        
    with col_d2:
        st.subheader("🎯 Aksi Rekomendasi Pemasaran")
        st.success(info_c["strategi"])
        
        st.markdown("""
        **Langkah Implementasi Tenant:**
        1. **Segmentasi Kampanye:** Sesuaikan saluran promosi (SMS, Email, atau Iklan Media Sosial).
        2. **Penentuan Harga & Produk:** Tampilkan etalase produk yang sesuai dengan daya beli segmen.
        3. **Evaluasi Penjualan:** Pantau rasio konversi promo dalam rentang waktu bulanan.
        """)
