import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# -------------------------------
# Keras 3 Compatibility Patch
# -------------------------------
from tensorflow.keras.layers import InputLayer
original_init = InputLayer.__init__

def patched_init(self, *args, **kwargs):
    if 'batch_shape' in kwargs and 'shape' not in kwargs:
        batch_shape = kwargs.pop('batch_shape')
        if batch_shape and batch_shape[0] is None:
            kwargs['shape'] = batch_shape[1:]
        else:
            kwargs['shape'] = batch_shape
    original_init(self, *args, **kwargs)

InputLayer.__init__ = patched_init

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AgriCure AI",
    page_icon="🌿",
    layout="wide"
)

# -------------------------------
# Styling
# -------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
    color: white;
}
.glass-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 20px;
}
.prediction-highlight {
    font-size: 26px;
    font-weight: bold;
    color: #00ffcc;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Helpers
# -------------------------------
@st.cache_resource
def load_prediction_model():
    path = "model/trained_model_new.keras"
    if not os.path.exists(path):
        path = "model/trained_model.keras"
    return tf.keras.models.load_model(path)

def load_metadata():
    with open("data/class_names.txt", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines()]

    with open("data/disease_info.json", encoding="utf-8") as f:
        info = json.load(f)

    return classes, info

def preprocess_image(img):
    img = img.resize((128, 128))
    arr = np.array(img)
    return np.expand_dims(arr, axis=0)

def get_text(details, field, lang):
    if lang == "Hindi":
        return details.get(f"{field}_hi", details.get(f"{field}_en", ""))
    return details.get(f"{field}_en", "")

# -------------------------------
# MAIN APP
# -------------------------------
def main():

    # Sidebar
    with st.sidebar:
        st.title("🌿 AgriCure")

        language = st.selectbox(
            "🌐 Language / भाषा",
            ["English", "Hindi"]
        )

        st.markdown("---")
        st.info("Upload a leaf image to detect disease.")

    # Labels
    if language == "Hindi":
        upload_label = "📸 पत्ते की छवि अपलोड करें"
        analyze_btn = "🔍 जांच करें"
        result_title = "🔍 परिणाम"
        desc_label = "📖 रोग विवरण"
        cure_label = "🛡️ उपचार"
        confidence_label = "विश्वास स्तर"
    else:
        upload_label = "📸 Upload Leaf Image"
        analyze_btn = "🔍 Analyze Plant"
        result_title = "🔍 Analysis Results"
        desc_label = "📖 Description"
        cure_label = "🛡️ Treatment"
        confidence_label = "Confidence Level"

 # Header
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    if language == "Hindi":
        st.title("🌿 पौधे की पत्ती की बीमारी की पहचान (एआई द्वारा)")
        st.write("पत्ते की फोटो अपलोड करें और तुरंत बीमारी की जानकारी पाएं।")
    else:
        st.title("🌿 AI Plant Leaf Disease Detection")
        st.write("Upload a leaf image and get instant diagnosis.")

    st.markdown('</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # LEFT SIDE
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(upload_label)

        file = st.file_uploader("Choose image", type=["jpg", "png", "jpeg"])

        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, use_container_width=True)

            if st.button(analyze_btn):
                with st.spinner("Analyzing..."):
                    model = load_prediction_model()
                    classes, info = load_metadata()

                    img = preprocess_image(image)
                    preds = model.predict(img)

                    idx = np.argmax(preds)
                    confidence = float(np.max(preds) * 100)
                    disease = classes[idx]

                    st.session_state["prediction"] = {
                        "name": disease,
                        "confidence": confidence,
                        "details": info.get(disease, {
                            "name_en": disease,
                            "name_hi": disease,
                            "desc_en": "No description available",
                            "desc_hi": "कोई विवरण उपलब्ध नहीं",
                            "cure_en": "No cure available",
                            "cure_hi": "कोई उपचार उपलब्ध नहीं"
                        })
                    }

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT SIDE
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader(result_title)

        if "prediction" in st.session_state:
            pred = st.session_state["prediction"]
            details = pred["details"]

            name = get_text(details, "name", language)
            desc = get_text(details, "desc", language)
            cure = get_text(details, "cure", language)

            st.markdown(f"<div class='prediction-highlight'>{name}</div>", unsafe_allow_html=True)
            st.progress(pred["confidence"] / 100)
            st.write(f"**{confidence_label}: {pred['confidence']:.2f}%**")

            st.write(f"### {desc_label}")
            st.write(desc)

            st.write(f"### {cure_label}")
            st.write(cure)

            if "healthy" in pred["name"].lower():
                st.balloons()
        else:
            st.info("Upload image to see results.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<center>AgriCure AI 🌿</center>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()