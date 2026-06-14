import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Membaca file dataset
# Pastikan file "Klasifikasi Tingkat Kemiskinan.csv" berada di folder yang sama dengan script Anda
df = pd.read_csv('Klasifikasi Tingkat Kemiskinan.csv')

# 2. Transformasi Data & Pemilihan Variabel sesuai Judul
# Mengubah PDRB ke bentuk Logaritma Alami (Log_PDRB) karena skalanya berupa miliaran/triliunan rupiah
df['Log_PDRB'] = np.log(df['PDRB atas Dasar Harga Konstan menurut Pengeluaran (Rupiah)'])

# Menentukan Variabel Independen (X) dan Dependen (Y)
X_cols = [
    'Tingkat Pengangguran Terbuka', 
    'Rata-rata Lama Sekolah Penduduk 15+ (Tahun)', 
    'Log_PDRB'
]
Y_col = 'Persentase Penduduk Miskin (P0) Menurut Kabupaten/Kota (Persen)'

X = df[X_cols]
Y = df[Y_col]

# Menambahkan konstanta (intercept) ke model regresi
X_with_constant = sm.add_constant(X)

# 3. Menjalankan Analisis Regresi OLS (Ordinary Least Squares)
model = sm.OLS(Y, X_with_constant).fit()

# 4. Menampilkan Ringkasan Hasil Analisis Statistik
print("="*60)
print("HASIL ANALISIS REGRESI LINEAR BERGANDA")
print("="*60)
print(model.summary())
print("="*60)

# 5. Membuat Visualisasi Evaluasi Model (Aktual vs Prediksi)
df['Prediksi_Kemiskinan'] = model.predict(X_with_constant)

plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")
sns.scatterplot(data=df, x='Prediksi_Kemiskinan', y=Y_col, alpha=0.6, color='royalblue')

# Membuat garis diagonal acuan (Jika prediksi 100% sempurna, semua titik berada di garis ini)
max_val = max(df[Y_col].max(), df['Prediksi_Kemiskinan'].max())
min_val = min(df[Y_col].min(), df['Prediksi_Kemiskinan'].min())
plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Garis Prediksi Sempurna')

plt.title('Plot Aktual vs Prediksi Persentase Penduduk Miskin\n(Kabupaten/Kota di Indonesia)', fontsize=13, fontweight='bold')
plt.xlabel('Nilai Prediksi Model (Persen)', fontsize=11)
plt.ylabel('Nilai Aktual Data Lapangan (Persen)', fontsize=11)
plt.legend()
plt.tight_layout()

# Menyimpan grafik ke file gambar
plt.savefig('aktual_vs_prediksi.png', dpi=300)
plt.show()