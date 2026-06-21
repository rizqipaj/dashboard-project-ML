# Dashboard Deteksi Emosi Wajah (FER2013)

## Struktur Folder
```
emotion-dashboard/
├── app.py              # Dashboard utama Streamlit
├── utils.py             # Preprocessing, deteksi wajah, prediksi
├── requirements.txt
├── model_emosi.h5       # <-- TARUH FILE MODEL HASIL TRAINING DI SINI
└── README.md
```

## Langkah Setup

### 1. Taruh File Model
Copy file `model_emosi.h5` (hasil `model.save()` dari notebook kamu) ke dalam folder ini,
sejajar dengan `app.py`.

### 2. PENTING: Cek Urutan Label Kelas
Di notebook training, jalankan:
```python
print(train_ds_raw.class_names)
```
Lalu buka `utils.py`, cek variabel `CLASS_NAMES` — urutannya HARUS SAMA PERSIS dengan output di atas.
Kalau urutan beda, hasil prediksi akan salah label walau model-nya benar.

### 3. Install Dependency
```bash
pip install -r requirements.txt
```

### 4. Jalankan Dashboard Secara Lokal
```bash
streamlit run app.py
```
Browser akan terbuka otomatis di `http://localhost:8501`

## Deployment (Gratis)

### Opsi A: Streamlit Community Cloud (paling mudah)
1. Push folder ini ke GitHub repository
2. Buka https://share.streamlit.io
3. Connect ke repo GitHub kamu, pilih `app.py` sebagai entry point
4. Deploy — selesai, dapat link publik

### Opsi B: Hugging Face Spaces
1. Buat akun di https://huggingface.co
2. New Space → pilih SDK "Streamlit"
3. Upload semua file (app.py, utils.py, requirements.txt, model_emosi.h5)
4. Space otomatis build dan jalan

## Catatan
- Kalau file model > 100MB, GitHub akan menolak push biasa — gunakan Git LFS,
  atau untuk Hugging Face Spaces bisa upload langsung lewat web UI (limitnya lebih besar).
- Format `.h5` maupun `.keras` sama-sama bisa di-load dengan `tf.keras.models.load_model()`,
  cukup sesuaikan nama file di `app.py` baris `load_model()`.
