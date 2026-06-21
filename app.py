"""
app.py
Dashboard deteksi emosi dari gambar wajah menggunakan model DCNN (FER2013).
Jalankan dengan: streamlit run app.py
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import cv2
import pandas as pd
import gdown
import os

from utils import (
    detect_face,
    preprocess_image,
    predict_emotion,
    pil_to_cv2,
    EMOJI_MAP,
)

# ===========================
# KONFIGURASI HALAMAN
# ===========================
st.set_page_config(
    page_title="Deteksi Emosi Wajah",
    page_icon="😊",
    layout="centered"
)


# ===========================
# LOAD MODEL (cached, hanya dijalankan sekali)
# ===========================
# ===========================
# DOWNLOAD MODEL DARI GOOGLE DRIVE (jika belum ada di lokal)
# ===========================
MODEL_PATH = "model_emosi.keras"
GDRIVE_FILE_ID = "1MR1c91tttSPjpGifFvdU3SvNBl7AysT"  # <-- ganti ini dengan ID file dari link share Drive

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Mengunduh model untuk pertama kali, mohon tunggu..."):
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


# ===========================
# HEADER
# ===========================
st.title("😊 Dashboard Deteksi Emosi Wajah")
st.markdown(
    "Upload foto yang menampilkan wajah, dan sistem akan mendeteksi ekspresi emosi "
    "menggunakan model **DCNN yang dilatih pada dataset FER2013**."
)
st.divider()


# ===========================
# UPLOAD GAMBAR
# ===========================
uploaded_file = st.file_uploader(
    "Upload gambar (JPG/PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    image_cv2 = pil_to_cv2(pil_image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(pil_image, use_container_width=True)

    # Deteksi wajah
    face_crop, bbox = detect_face(image_cv2)

    if face_crop is None:
        st.error(
            "⚠️ Wajah tidak terdeteksi pada gambar ini. "
            "Coba upload foto dengan wajah yang lebih jelas dan menghadap ke depan."
        )
    else:
        with col2:
            st.subheader("Wajah Terdeteksi")
            x, y, w, h = bbox
            image_with_box = image_cv2.copy()
            cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image_with_box_rgb = cv2.cvtColor(image_with_box, cv2.COLOR_BGR2RGB)
            st.image(image_with_box_rgb, use_container_width=True)

        st.divider()

        # Preprocessing & prediksi
        face_input = preprocess_image(face_crop)
        label, confidence, all_probs = predict_emotion(model, face_input)

        # ===========================
        # HASIL PREDIKSI
        # ===========================
        st.subheader("Hasil Deteksi")

        emoji = EMOJI_MAP.get(label, "")
        st.markdown(f"## {emoji} **{label}**")
        st.progress(confidence, text=f"Confidence: {confidence * 100:.2f}%")

        st.divider()

        # ===========================
        # GRAFIK PROBABILITAS SEMUA KELAS
        # ===========================
        st.subheader("Distribusi Probabilitas Tiap Emosi")

        df_probs = pd.DataFrame({
            "Emosi": list(all_probs.keys()),
            "Probabilitas": list(all_probs.values())
        }).sort_values("Probabilitas", ascending=False)

        st.bar_chart(df_probs.set_index("Emosi"))
        st.dataframe(
            df_probs.style.format({"Probabilitas": "{:.2%}"}),
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("👆 Upload gambar untuk mulai deteksi emosi.")

st.divider()
st.caption("Model: DCNN (Conv2D + BatchNorm) dilatih pada dataset FER2013 | Dashboard dibuat dengan Streamlit")
