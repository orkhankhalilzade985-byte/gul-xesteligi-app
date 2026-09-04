import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

# ---------- Səhifə ayarları ----------
st.set_page_config(
    page_title="Agro Food Investments | Gül Xəstəliyi Aşkarlama",
    page_icon="🌹",
    layout="centered"
)

# ---------- Xüsusi CSS ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    .block-container {
        padding-top: 1.2rem;
        max-width: 780px;
    }

    /* --- HERO KARTI --- */
    .hero-card {
        background: linear-gradient(135deg, #1b4d3e 0%, #2e7d5b 100%);
        border-radius: 18px;
        padding: 2rem 1.5rem 1.6rem 1.5rem;
        text-align: center;
        box-shadow: 0 8px 24px rgba(27, 77, 62, 0.25);
        margin-bottom: 1.8rem;
    }
    .hero-title {
        color: white;
        font-size: 1.9rem;
        font-weight: 700;
        margin: 0.8rem 0 0.2rem 0;
    }
    .hero-sub {
        color: #d7ecdf;
        font-size: 1rem;
        margin-bottom: 0.9rem;
    }
    .creator-pill {
        display: inline-block;
        background-color: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.35);
        color: white;
        padding: 5px 16px;
        border-radius: 20px;
        font-size: 0.82rem;
    }

    /* --- MƏLUMAT KARTI --- */
    .info-card {
        background-color: #eef6ee;
        border-left: 4px solid #2e7d5b;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
        font-size: 0.95rem;
        color: #1f2b23;
    }

    /* --- Yükləmə qutusu --- */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #fafffa;
        border: 2px dashed #2e7d5b;
        border-radius: 14px;
    }

    /* --- Nəticə kartları --- */
    .result-card {
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        color: white;
        font-size: 1.05rem;
        font-weight: 600;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }
    .result-high   { background: linear-gradient(135deg, #2e7d5b, #1b4d3e); }
    .result-medium { background: linear-gradient(135deg, #d98e28, #b56f16); }
    .result-low    { background: linear-gradient(135deg, #c0392b, #922b21); }

    .stProgress > div > div > div > div {
        background-color: #2e7d5b;
    }

    /* --- Footer --- */
    .footer-bar {
        background-color: #1b4d3e;
        color: #d7ecdf;
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        margin-top: 2.5rem;
        font-size: 0.82rem;
    }
    .footer-bar b { color: white; }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------- HERO bölməsi ----------
logo_html = ""
if os.path.exists("logo_clean.png"):
    import base64
    with open("logo_clean.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    logo_html = f'<img src="data:image/png;base64,{logo_b64}" width="120">'

st.markdown(f"""
<div class="hero-card">
    {logo_html}
    <div class="hero-title">🌹 Qızılgül Yarpağı Xəstəlik Aşkarlama</div>
    <div class="hero-sub">Agro Food Investments — Süni İntellekt Əsaslı Diaqnostika Sistemi</div>
    <span class="creator-pill">👨‍💻 Created by Orkhan Khalilzada</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-card">
Bu sistem <b>Transfer Learning (MobileNetV2)</b> əsasında qurulmuş dərin öyrənmə modelidir.
Qızılgül yarpağının şəklini yükləyin — model onun <b>sağlam</b>, <b>pas xəstəliyi (Rust)</b>,
yoxsa <b>sawfly həşərat zədəsi</b> olduğunu təyin etsin.
</div>
""", unsafe_allow_html=True)

# ---------- Model yükləmə ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model_v2.keras")

model = load_model()

CLASS_NAMES = ['Healthy_Leaf_Rose', 'Rose_Rust', 'Rose_sawfly_Rose_slug']
CLASS_NAMES_AZ = {
    'Healthy_Leaf_Rose': '🟢 Sağlam Yarpaq',
    'Rose_Rust': '🟠 Qızılgül Pası (Rust)',
    'Rose_sawfly_Rose_slug': '🟣 Sawfly Həşərat Zədəsi'
}

def remove_background(img_array):
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lower = np.array([15, 30, 30])
    upper = np.array([95, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_and(img_array, img_array, mask=mask)

# ---------- Şəkil yükləmə ----------
uploaded_file = st.file_uploader("📤 Qızılgül yarpağı şəklini yükləyin", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(pil_img)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Yüklənən şəkil**")
        st.image(pil_img, use_container_width=True)

    img_no_bg = remove_background(img_array)
    with col2:
        st.markdown("**Emal edilmiş (fonsuz)**")
        st.image(img_no_bg, use_container_width=True)

    img_resized = cv2.resize(img_no_bg, (224, 224))
    img_batch = np.expand_dims(img_resized, axis=0)

    with st.spinner("Model təhlil edir..."):
        predictions = model.predict(img_batch, verbose=0)[0]

    predicted_idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = predictions[predicted_idx] * 100

    st.markdown("### 📋 Nəticə")

    if confidence > 70:
        card_class = "result-high"
    elif confidence > 45:
        card_class = "result-medium"
    else:
        card_class = "result-low"

    st.markdown(f"""
    <div class="result-card {card_class}">
        {CLASS_NAMES_AZ[predicted_class]} — {confidence:.1f}% əminliklə
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Ətraflı ehtimallar:**")
    for i, cls in enumerate(CLASS_NAMES):
        st.progress(float(predictions[i]), text=f"{CLASS_NAMES_AZ[cls]}: {predictions[i]*100:.1f}%")

    st.caption("⚠️ Bu model tədqiqat/demo məqsədlidir. Hazırda yalnız 3 halı tanıyır: sağlam, pas xəstəliyi və sawfly zədəsi.")
else:
    st.info("👆 Başlamaq üçün bir qızılgül yarpağı şəkli yükləyin")

# ---------- Footer ----------
st.markdown("""
<div class="footer-bar">
    Agro Food Investments &copy; 2026 &nbsp;|&nbsp; Created by <b>Orkhan Khalilzada</b> &nbsp;|&nbsp; MobileNetV2 Transfer Learning
</div>
""", unsafe_allow_html=True)
