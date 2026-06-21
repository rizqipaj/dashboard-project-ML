"""
utils.py
Fungsi-fungsi bantuan untuk preprocessing gambar dan prediksi emosi.
"""

import numpy as np
import cv2
from PIL import Image
import tensorflow as tf

# ===========================
# KONFIGURASI
# ===========================
IMG_SIZE = (48, 48)

# PENTING: Urutan label HARUS sama persis dengan urutan class_names saat training.
# Default image_dataset_from_directory mengurutkan folder secara alfabetis, biasanya:
# ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Cek ulang dengan print(train_ds_raw.class_names) di notebook kamu, lalu sesuaikan di sini kalau berbeda.
CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

EMOJI_MAP = {
    'Angry': '😠',
    'Disgust': '🤢',
    'Fear': '😨',
    'Happy': '😄',
    'Neutral': '😐',
    'Sad': '😢',
    'Surprise': '😲',
}

# Haar Cascade untuk deteksi wajah (sudah include bawaan OpenCV, tidak perlu download manual)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_face(image_np):
    """
    Mendeteksi wajah pada gambar (numpy array, BGR atau grayscale).
    Mengembalikan crop wajah grayscale jika ditemukan, atau None jika tidak ada wajah.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    if len(faces) == 0:
        return None, None

    # Ambil wajah dengan area terbesar (asumsi wajah utama/paling dekat ke kamera)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = gray[y:y + h, x:x + w]
    bbox = (x, y, w, h)

    return face_crop, bbox


def preprocess_image(face_crop):
    """
    Preprocessing crop wajah grayscale menjadi input siap pakai untuk model.
    Output shape: (1, 48, 48, 1), dinormalisasi 0-1.
    """
    resized = cv2.resize(face_crop, IMG_SIZE)
    normalized = resized.astype('float32') / 255.0
    reshaped = normalized.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)
    return reshaped


def predict_emotion(model, face_input):
    """
    Melakukan prediksi emosi dari input yang sudah dipreprocess.
    Mengembalikan label emosi, confidence, dan dictionary semua probabilitas.
    """
    predictions = model.predict(face_input, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    label = CLASS_NAMES[pred_idx]
    confidence = float(predictions[pred_idx])

    all_probs = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

    return label, confidence, all_probs


def pil_to_cv2(pil_image):
    """Konversi gambar PIL (hasil upload Streamlit) ke format numpy array BGR untuk OpenCV."""
    image_rgb = np.array(pil_image.convert('RGB'))
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr
