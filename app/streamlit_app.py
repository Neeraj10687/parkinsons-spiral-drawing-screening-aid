"""
Streamlit demo for Parkinson's Spiral Drawing Screening Aid.

Supports both models:
  - Handcrafted Features + Logistic Regression (interpretable baseline)
  - CNN (3-block neural network, v4 with oversampling)

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_newhandpd
from src.features import extract_all_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Feature info
# =============================================================================
FEATURE_INFO = {
    "radius_deviation": {
        "title": "Spiral Shape Irregularity",
        "plain": "How far the actual spiral deviates from a perfect, smooth Archimedean spiral.",
        "direction": "higher",
        "cue": "\u2191 Higher = more irregular",
        "decimals": 3,
    },
    "stroke_smoothness": {
        "title": "Stroke Jerkiness",
        "plain": "How jerky the hand movement was \u2014 sudden changes in pen direction.",
        "direction": "higher",
        "cue": "\u2191 Higher = jerkier drawing",
        "decimals": 4,
    },
    "stroke_entropy": {
        "title": "Direction Randomness",
        "plain": "How random / unpredictable the stroke direction is.",
        "direction": "higher",
        "cue": "\u2191 Higher = more random",
        "decimals": 4,
    },
    "intersection_count": {
        "title": "Self-Intersections",
        "plain": "How many times the spiral crosses itself.",
        "direction": "higher",
        "cue": "\u2191 Higher = more chaotic",
        "decimals": 0,
    },
    "mean_squared_displacement": {
        "title": "Spiral Spread",
        "plain": "How wide the spiral is \u2014 small spirals are a classic PD sign (micrographia).",
        "direction": "lower",
        "cue": "\u2193 Lower = smaller spiral = PD marker",
        "decimals": 2,
    },
}

FEATURE_ORDER = [
    "mean_squared_displacement",
    "stroke_smoothness",
    "stroke_entropy",
    "intersection_count",
    "radius_deviation",
]

# --- Page Config ---
st.set_page_config(page_title="Parkinson's Spiral Screening", page_icon="\U0001F9E0", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1A5276 0%, #2E86C1 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .main-header h1 {
        color: white;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        color: rgba(255,255,255,0.8);
        font-size: 0.95rem;
        margin: 4px 0 0 0;
    }
    .warning-banner {
        background: #FDEBD0;
        border-left: 4px solid #E67E22;
        padding: 10px 16px;
        border-radius: 6px;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }
    .pred-card {
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 12px;
    }
    .pred-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .pred-conf {
        font-size: 0.9rem;
        margin-bottom: 8px;
    }
    .conf-bar-bg {
        background: #FFFFFF;
        border-radius: 6px;
        height: 12px;
        overflow: hidden;
        border: 1px solid rgba(0,0,0,0.1);
    }
    .conf-bar-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.3s;
    }
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 8px;
    }
    .feature-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1A5276;
    }
    .feature-value {
        font-family: monospace;
        font-size: 1.4rem;
        font-weight: 700;
        color: #1A5276;
        text-align: right;
    }
    .stButton > button {
        border-radius: 8px;
        border: 2px solid #2E86C1;
        color: #2E86C1;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background: #2E86C1;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown(
    '<div class="main-header">'
    '<h1>\U0001F9E0 Parkinson\u2019s Spiral Drawing Screening Aid</h1>'
    '<p>LR Baseline + CNN Comparison \u00b7 Interpretable ML Demo</p>'
    '</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="warning-banner">'
    '<strong>\u26a0\ufe0f Screening aid, not a diagnostic device.</strong> '
    'This tool runs two models on your spiral image and compares their predictions. '
    'For research / educational demonstration only.'
    '</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# Load models
# =============================================================================
@st.cache_resource
def load_lr_model():
    images, labels, patient_ids, filenames = load_newhandpd("data/raw/Merged")
    features = np.load("data/processed/features_handcrafted.npy")
    feature_names = np.load("data/processed/feature_names.npy", allow_pickle=True).tolist()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(features_scaled, labels)
    return images, labels, patient_ids, filenames, model, scaler, feature_names


@st.cache_resource
def load_cnn_model():
    import tensorflow as tf
    from src.model import build_model
    model = build_model()
    weights_path = "data/processed/cnn_model_weights.weights.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        return model, True
    return None, False


try:
    images, labels, patient_ids, filenames, lr_model, scaler, feature_names = load_lr_model()
except Exception as e:
    st.error(f"Error loading LR model: {e}")
    st.info("Run `python src/features.py` and `python src/train.py --model baseline` first.")
    st.stop()

try:
    cnn_model, cnn_available = load_cnn_model()
except Exception:
    cnn_model = None
    cnn_available = False


@st.cache_data
def compute_reference_stats():
    features = np.load("data/processed/features_handcrafted.npy")
    feature_names = np.load("data/processed/feature_names.npy", allow_pickle=True).tolist()
    stats = {}
    for i, name in enumerate(feature_names):
        col = features[:, i]
        stats[name] = {"min": float(np.min(col)), "max": float(np.max(col)), "median": float(np.median(col))}
    return stats

try:
    REF_STATS = compute_reference_stats()
except Exception:
    REF_STATS = None


# =============================================================================
# Sidebar
# =============================================================================
st.sidebar.markdown("### \U0001F3AF Input")
input_method = st.sidebar.radio("Choose:", ["Upload Image", "Random from Dataset"], label_visibility="collapsed")

if cnn_available:
    st.sidebar.markdown("\u2705 **CNN loaded**")
else:
    st.sidebar.markdown("\u26a0\ufe0f **CNN not loaded**")

image_to_test = None
true_label = None
true_filename = None

if input_method == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload spiral (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            image_to_test = img
            true_filename = uploaded_file.name
else:
    if st.sidebar.button("\U0001F3B2 Pick Random Image"):
        idx = np.random.randint(0, len(images))
        image_to_test = images[idx]
        true_label = labels[idx]
        true_filename = filenames[idx]
        st.session_state['random_idx'] = idx
    elif 'random_idx' in st.session_state:
        idx = st.session_state['random_idx']
        image_to_test = images[idx]
        true_label = labels[idx]
        true_filename = filenames[idx]


# =============================================================================
# Main content
# =============================================================================
if image_to_test is not None:
    # --- Run LR ---
    with st.spinner("Extracting features..."):
        features_dict = extract_all_features(image_to_test)
        fv = np.array([features_dict[n] for n in feature_names]).reshape(1, -1)
        fv_s = scaler.transform(fv)
        lr_prob = float(lr_model.predict_proba(fv_s)[0, 1])
        lr_pred = 1 if lr_prob >= 0.5 else 0

    # --- Run CNN ---
    cnn_prob = None
    cnn_pred = None
    if cnn_available and cnn_model is not None:
        with st.spinner("Running CNN..."):
            img_batch = np.expand_dims(image_to_test, axis=0)
            cnn_prob = float(cnn_model.predict(img_batch, verbose=0)[0, 0])
            cnn_pred = 1 if cnn_prob >= 0.5 else 0

    # ======= Row 1: Image + Predictions =======
    col_img, col_pred = st.columns([1, 1.4])

    with col_img:
        st.markdown("#### \U0001F5BC\ufe0f Spiral Image")
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.imshow(image_to_test.squeeze(), cmap="gray")
        ax.axis("off")
        if true_filename:
            ax.set_title(true_filename, fontsize=9, color="#5B6B7D")
        st.pyplot(fig)

        if true_label is not None:
            ls = "Parkinson's" if true_label == 1 else "Healthy"
            lc = "#C0392B" if true_label == 1 else "#1E8449"
            st.markdown(
                f'<div style="background:#EDF3F8;padding:8px 14px;border-radius:8px;'
                f'border-left:4px solid {lc};font-size:0.9rem;">'
                f'<strong>True label:</strong> <span style="color:{lc};font-weight:700;">{ls}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with col_pred:
        st.markdown("#### \U0001F9E0 Predictions")

        # --- LR card ---
        lr_bg = "#FDEDEC" if lr_pred == 1 else "#E8F5E9"
        lr_bd = "#C0392B" if lr_pred == 1 else "#1E8449"
        lr_tc = "#922B21" if lr_pred == 1 else "#1E8449"
        lr_ic = "\u26a0\ufe0f" if lr_pred == 1 else "\u2705"
        lr_rt = "Parkinson's-screening-positive" if lr_pred == 1 else "Healthy"
        lr_pct = lr_prob * 100

        st.markdown(
            f'<div class="pred-card" style="background:{lr_bg};border:2px solid {lr_bd};">'
            f'<div class="pred-title" style="color:{lr_tc};">'
            f'\U0001F4CA LR Baseline: {lr_ic} {lr_rt}</div>'
            f'<div class="pred-conf" style="color:#1A2B3C;">'
            f'Confidence: <strong>{lr_pct:.1f}%</strong> chance of PD '
            f'<span style="color:#5B6B7D;font-size:0.8rem;">(5 handcrafted features)</span></div>'
            f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="background:{lr_bd};width:{lr_pct:.1f}%;"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # --- CNN card ---
        if cnn_prob is not None:
            cnn_bg = "#FDEDEC" if cnn_pred == 1 else "#E8F5E9"
            cnn_bd = "#C0392B" if cnn_pred == 1 else "#1E8449"
            cnn_tc = "#922B21" if cnn_pred == 1 else "#1E8449"
            cnn_ic = "\u26a0\ufe0f" if cnn_pred == 1 else "\u2705"
            cnn_rt = "Parkinson's-screening-positive" if cnn_pred == 1 else "Healthy"
            cnn_pct = cnn_prob * 100

            st.markdown(
                f'<div class="pred-card" style="background:{cnn_bg};border:2px solid {cnn_bd};">'
                f'<div class="pred-title" style="color:{cnn_tc};">'
                f'\U0001F9E0 CNN Model: {cnn_ic} {cnn_rt}</div>'
                f'<div class="pred-conf" style="color:#1A2B3C;">'
                f'Confidence: <strong>{cnn_pct:.1f}%</strong> chance of PD '
                f'<span style="color:#5B6B7D;font-size:0.8rem;">(3-block neural network)</span></div>'
                f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="background:{cnn_bd};width:{cnn_pct:.1f}%;"></div></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Agreement
            if lr_pred == cnn_pred:
                st.markdown(
                    f'<div style="background:#E8F5E9;padding:10px 16px;border-radius:8px;'
                    f'border-left:4px solid #1E8449;font-size:0.9rem;">'
                    f'\u2705 <strong>Both models agree</strong> \u2014 prediction: {lr_rt}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div style="background:#FEF9E7;padding:10px 16px;border-radius:8px;'
                    f'border-left:4px solid #F1C40F;font-size:0.9rem;">'
                    f'\u26a0\ufe0f <strong>Models disagree</strong> \u2014 LR says {("PD" if lr_pred else "Healthy")}, '
                    f'CNN says {("PD" if cnn_pred else "Healthy")}</div>',
                    unsafe_allow_html=True,
                )

            # True label match
            if true_label is not None:
                lr_ok = "\u2705" if lr_pred == true_label else "\u274c"
                cnn_ok = "\u2705" if cnn_pred == true_label else "\u274c"
                st.markdown(
                    f'<div style="margin-top:8px;font-size:0.85rem;color:#5B6B7D;">'
                    f'Verification: LR {lr_ok} \u00b7 CNN {cnn_ok}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="pred-card" style="background:#F5F8FC;border:2px dashed #D6E0EA;">'
                '<div class="pred-title" style="color:#5B6B7D;">\U0001F9E0 CNN: Not available</div>'
                '<div style="font-size:0.85rem;color:#5B6B7D;">'
                'Save weights to <code>data/processed/cnn_model_weights.weights.h5</code></div>'
                '</div>',
                unsafe_allow_html=True,
            )

    # ======= Feature breakdown =======
    st.markdown("---")
    st.markdown("### \U0001F50E The 5 Geometric Features")
    st.markdown(
        '<p style="color:#5B6B7D;font-size:0.9rem;">'
        'What the LR model measured. The CNN doesn\u2019t use these \u2014 it reads the raw pixels.</p>',
        unsafe_allow_html=True,
    )

    def pct_rank(value, name):
        if REF_STATS is None or name not in REF_STATS:
            return None
        s = REF_STATS[name]
        if s["max"] == s["min"]:
            return 50.0
        return ((value - s["min"]) / (s["max"] - s["min"])) * 100.0

    cols = st.columns(5)
    for i, name in enumerate(FEATURE_ORDER):
        if name not in features_dict:
            continue
        info = FEATURE_INFO[name]
        value = features_dict[name]
        pct = pct_rank(value, name)

        if info["decimals"] == 0:
            vs = f"{int(round(value))}"
        else:
            vs = f"{value:.{info['decimals']}f}"

        dc = "#C0392B" if info["direction"] == "higher" else "#1E8449"

        with cols[i]:
            st.markdown(
                f'<div class="feature-card" style="border-left:4px solid {dc};">'
                f'<div class="feature-title">{info["title"]}</div>'
                f'<div style="font-size:0.8rem;color:#5B6B7D;margin:4px 0;">{info["plain"]}</div>'
                f'<div class="feature-value">{vs}</div>'
                f'<div style="font-size:0.8rem;color:{dc};font-weight:600;margin-top:4px;">{info["cue"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

else:
    st.markdown(
        '<div style="background:#EDF3F8;border:2px dashed #2E86C1;border-radius:12px;'
        'padding:48px;text-align:center;margin-top:30px;">'
        '<div style="font-size:3rem;margin-bottom:12px;">\U0001F448</div>'
        '<div style="font-size:1.2rem;color:#1A5276;font-weight:700;margin-bottom:8px;">'
        'Upload a spiral image to begin</div>'
        '<div style="color:#5B6B7D;font-size:0.95rem;">'
        'Upload your own PNG/JPG or pick a random one from the dataset</div>'
        '</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "\U0001F9E0 Parkinson\u2019s Spiral Drawing Screening Aid \u00b7 LR + CNN \u00b7 "
    "Educational demo, not a clinical diagnostic tool."
)
