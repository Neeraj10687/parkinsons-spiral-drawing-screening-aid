"""
Streamlit demo for Parkinson's Spiral Drawing Screening Aid.

Supports four models:
  1. Handcrafted Features + Logistic Regression (interpretable baseline)
  2. CNN (3-block, oversampling)
  3. MobileNetV2 (transfer learning)
  4. Ensemble (average of all 3)

Two pages:
  1. Screening Demo — upload/pick image, see all model predictions + features
  2. Model Details — architecture, metrics, training history for all models

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

# =============================================================================
# Model metrics (from Colab training results)
# =============================================================================
MODEL_METRICS = {
    "lr": {"name": "LR Baseline", "accuracy": 0.788, "recall": 0.802, "precision": 0.869, "f1": 0.834,
           "acc_ci": "[0.755, 0.821]", "params": "6 parameters", "color": "#1E8449"},
    "cnn": {"name": "CNN v4", "accuracy": 0.815, "recall": 0.979, "precision": 0.830, "f1": 0.880,
            "acc_ci": "[0.783, 0.845]", "params": "~110K parameters", "color": "#2E86C1"},
    "mobilenet": {"name": "MobileNetV2", "accuracy": 0.894, "recall": 0.936, "precision": 0.908, "f1": 0.921,
                  "acc_ci": "[0.870, 0.916]", "params": "~3.5M (frozen) + 8K trainable", "color": "#E67E22"},
    "ensemble": {"name": "Ensemble (3-way)", "accuracy": 0.929, "recall": 0.988, "precision": 0.912, "f1": 0.949,
                 "acc_ci": "[0.910, 0.948]", "params": "Average of LR + CNN + MobileNetV2", "color": "#8E44AD"},
}

MCNEMAR_RESULTS = {
    "CNN vs LR": "p=0.249 (not significant)",
    "MobileNetV2 vs LR": "p<0.001 (SIGNIFICANT)",
    "MobileNetV2 vs CNN": "p<0.001 (SIGNIFICANT)",
    "Ensemble vs CNN": "p<0.001 (SIGNIFICANT)",
    "Ensemble vs MobileNetV2": "p=0.002 (SIGNIFICANT)",
}

# --- Page Config ---
st.set_page_config(page_title="Parkinson's Spiral Screening", page_icon="\U0001F9E0", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1A5276 0%, #2E86C1 100%);
        padding: 20px 30px; border-radius: 12px; margin-bottom: 20px;
    }
    .main-header h1 { color: white; font-size: 1.8rem; margin: 0; font-weight: 700; }
    .main-header p { color: rgba(255,255,255,0.8); font-size: 0.95rem; margin: 4px 0 0 0; }
    .warning-banner {
        background: #FDEBD0; border-left: 4px solid #E67E22;
        padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-size: 0.9rem;
    }
    .pred-card { border-radius: 12px; padding: 14px 18px; margin-bottom: 8px; }
    .pred-title { font-size: 1.05rem; font-weight: 700; margin-bottom: 6px; }
    .pred-conf { font-size: 0.85rem; margin-bottom: 6px; }
    .conf-bar-bg { background: #FFFFFF; border-radius: 6px; height: 10px; overflow: hidden; border: 1px solid rgba(0,0,0,0.1); }
    .conf-bar-fill { height: 100%; border-radius: 6px; }
    .feature-card { background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 10px; padding: 14px 18px; margin-bottom: 8px; }
    .feature-title { font-size: 1rem; font-weight: 700; color: #1A5276; }
    .feature-value { font-family: monospace; font-size: 1.4rem; font-weight: 700; color: #1A5276; text-align: right; }
    .metric-card { background: #FFFFFF; border: 1px solid #E0E0E0; border-radius: 10px; padding: 16px; text-align: center; }
    .metric-num { font-family: monospace; font-size: 1.6rem; font-weight: 700; }
    .metric-label { font-size: 0.75rem; color: #5B6B7D; margin-top: 4px; }
    .stButton > button { border-radius: 8px; border: 2px solid #2E86C1; color: #2E86C1; font-weight: 600; width: 100%; }
    .stButton > button:hover { background: #2E86C1; color: white; }
</style>
""", unsafe_allow_html=True)


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


@st.cache_resource
def load_mobilenet_model():
    import tensorflow as tf
    base = tf.keras.applications.MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
    base.trainable = False
    model = tf.keras.Sequential([base, tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')], name='spiral_mobilenetv2')
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    weights_path = "data/processed/mobilenet_weights.weights.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        return model, True
    return None, False


try:
    images, labels, patient_ids, filenames, lr_model, scaler, feature_names = load_lr_model()
except Exception as e:
    st.error(f"Error loading LR model: {e}")
    st.stop()

try:
    cnn_model, cnn_available = load_cnn_model()
except Exception:
    cnn_model, cnn_available = None, False

try:
    mobilenet_model, mobilenet_available = load_mobilenet_model()
except Exception:
    mobilenet_model, mobilenet_available = None, False


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
# Navigation
# =============================================================================
st.sidebar.markdown("### \U0001F3AF Navigation")
page = st.sidebar.radio("Page:", ["\U0001F5BC\ufe0f Screening Demo", "\U0001F4CA Model Details"], label_visibility="collapsed")

# Model status in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**Model Status:**")
st.sidebar.markdown(f"\u2705 LR Baseline (always available)")
if cnn_available:
    st.sidebar.markdown(f"\u2705 CNN loaded")
else:
    st.sidebar.markdown(f"\u274c CNN not loaded")
if mobilenet_available:
    st.sidebar.markdown(f"\u2705 MobileNetV2 loaded")
else:
    st.sidebar.markdown(f"\u274c MobileNetV2 not loaded")


# =============================================================================
# PAGE 1: Screening Demo
# =============================================================================
if page == "\U0001F5BC\ufe0f Screening Demo":

    st.markdown(
        '<div class="main-header"><h1>\U0001F9E0 Parkinson\u2019s Spiral Drawing Screening Aid</h1>'
        '<p>4-Model Comparison: LR + CNN + MobileNetV2 + Ensemble</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="warning-banner"><strong>\u26a0\ufe0f Screening aid, not a diagnostic device.</strong> '
        'For research / educational demonstration only.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar input
    st.sidebar.markdown("### \U0001F3AF Input")
    input_method = st.sidebar.radio("Choose:", ["Upload Image", "Random from Dataset"], label_visibility="collapsed")

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

        # --- Run MobileNetV2 ---
        mb_prob = None
        mb_pred = None
        if mobilenet_available and mobilenet_model is not None:
            with st.spinner("Running MobileNetV2..."):
                img_3ch = np.repeat(np.expand_dims(image_to_test, axis=0), 3, axis=-1)
                mb_prob = float(mobilenet_model.predict(img_3ch, verbose=0)[0, 0])
                mb_pred = 1 if mb_prob >= 0.5 else 0

        # --- Ensemble ---
        ens_prob = None
        ens_pred = None
        all_probs = [lr_prob]
        if cnn_prob is not None:
            all_probs.append(cnn_prob)
        if mb_prob is not None:
            all_probs.append(mb_prob)
        if len(all_probs) > 1:
            ens_prob = float(np.mean(all_probs))
            ens_pred = 1 if ens_prob >= 0.5 else 0

        # ======= Layout =======
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
                    f'<strong>True label:</strong> <span style="color:{lc};font-weight:700;">{ls}</span></div>',
                    unsafe_allow_html=True,
                )

        with col_pred:
            st.markdown("#### \U0001F9E0 Predictions")

            # Helper to render a prediction card
            def render_card(name, pred, prob, color, subtitle):
                bg = "#FDEDEC" if pred == 1 else "#E8F5E9"
                bd = "#C0392B" if pred == 1 else "#1E8449"
                tc = "#922B21" if pred == 1 else "#1E8449"
                ic = "\u26a0\ufe0f" if pred == 1 else "\u2705"
                rt = "PD-screening-positive" if pred == 1 else "Healthy"
                pct = prob * 100
                st.markdown(
                    f'<div class="pred-card" style="background:{bg};border:2px solid {bd};">'
                    f'<div class="pred-title" style="color:{tc};">{ic} {name}: {rt}</div>'
                    f'<div class="pred-conf" style="color:#1A2B3C;">'
                    f'Confidence: <strong>{pct:.1f}%</strong> '
                    f'<span style="color:#5B6B7D;font-size:0.8rem;">({subtitle})</span></div>'
                    f'<div class="conf-bar-bg"><div class="conf-bar-fill" style="background:{bd};width:{pct:.1f}%;"></div></div>'
                    f'</div>', unsafe_allow_html=True,
                )

            # LR card
            render_card("LR Baseline", lr_pred, lr_prob, "#1E8449", "5 handcrafted features")

            # CNN card
            if cnn_prob is not None:
                render_card("CNN v4", cnn_pred, cnn_prob, "#2E86C1", "3-block neural network")
            else:
                st.markdown('<div class="pred-card" style="background:#F5F8FC;border:2px dashed #D6E0EA;">'
                    '<div class="pred-title" style="color:#5B6B7D;">CNN: Not available</div></div>',
                    unsafe_allow_html=True)

            # MobileNetV2 card
            if mb_prob is not None:
                render_card("MobileNetV2", mb_pred, mb_prob, "#E67E22", "transfer learning (ImageNet)")
            else:
                st.markdown('<div class="pred-card" style="background:#F5F8FC;border:2px dashed #D6E0EA;">'
                    '<div class="pred-title" style="color:#5B6B7D;">MobileNetV2: Not available</div></div>',
                    unsafe_allow_html=True)

            # Ensemble card
            if ens_prob is not None:
                render_card("Ensemble", ens_pred, ens_prob, "#8E44AD", "average of all models")

            # Agreement check
            preds_list = [lr_pred]
            if cnn_prob is not None: preds_list.append(cnn_pred)
            if mb_prob is not None: preds_list.append(mb_pred)
            if ens_prob is not None: preds_list.append(ens_pred)

            if len(set(preds_list)) == 1:
                st.markdown(
                    f'<div style="background:#E8F5E9;padding:8px 14px;border-radius:8px;'
                    f'border-left:4px solid #1E8449;font-size:0.85rem;">'
                    f'\u2705 <strong>All models agree</strong></div>', unsafe_allow_html=True,
                )
            else:
                pd_count = sum(preds_list)
                hl_count = len(preds_list) - pd_count
                st.markdown(
                    f'<div style="background:#FEF9E7;padding:8px 14px;border-radius:8px;'
                    f'border-left:4px solid #F1C40F;font-size:0.85rem;">'
                    f'\u26a0\ufe0f <strong>Models split:</strong> {pd_count} say PD, {hl_count} say Healthy</div>',
                    unsafe_allow_html=True,
                )

            # Verification
            if true_label is not None:
                checks = []
                checks.append(("LR", "\u2705" if lr_pred == true_label else "\u274c"))
                if cnn_prob is not None:
                    checks.append(("CNN", "\u2705" if cnn_pred == true_label else "\u274c"))
                if mb_prob is not None:
                    checks.append(("MBV2", "\u2705" if mb_pred == true_label else "\u274c"))
                if ens_prob is not None:
                    checks.append(("Ens", "\u2705" if ens_pred == true_label else "\u274c"))
                ver_str = " \u00b7 ".join(f"{n} {c}" for n, c in checks)
                st.markdown(
                    f'<div style="margin-top:6px;font-size:0.8rem;color:#5B6B7D;">Verification: {ver_str}</div>',
                    unsafe_allow_html=True,
                )

        # Features
        st.markdown("---")
        st.markdown("### \U0001F50E The 5 Geometric Features (LR only)")
        st.markdown('<p style="color:#5B6B7D;font-size:0.9rem;">CNN and MobileNetV2 read raw pixels \u2014 they don\'t use these.</p>', unsafe_allow_html=True)

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
                    f'</div>', unsafe_allow_html=True,
                )

    else:
        st.markdown(
            '<div style="background:#EDF3F8;border:2px dashed #2E86C1;border-radius:12px;'
            'padding:48px;text-align:center;margin-top:30px;">'
            '<div style="font-size:3rem;margin-bottom:12px;">\U0001F448</div>'
            '<div style="font-size:1.2rem;color:#1A5276;font-weight:700;">Upload a spiral image to begin</div>'
            '<div style="color:#5B6B7D;font-size:0.95rem;">Upload PNG/JPG or pick a random one from the dataset</div>'
            '</div>', unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 2: Model Details
# =============================================================================
elif page == "\U0001F4CA Model Details":

    st.markdown(
        '<div class="main-header"><h1>\U0001F4CA Model Details</h1>'
        '<p>4 models compared \u00b7 Architecture, metrics, and statistical tests</p></div>',
        unsafe_allow_html=True,
    )

    # --- Performance comparison ---
    st.markdown("### \U0001F4C8 Performance Comparison (Patient-Level 5-Fold CV)")

    models_to_show = ["lr", "cnn", "mobilenet", "ensemble"]
    metric_cols = st.columns(4)

    metrics_display = [
        ("Accuracy", "accuracy"),
        ("Recall", "recall"),
        ("Precision", "precision"),
        ("F1 Score", "f1"),
    ]

    for mi, (label, key) in enumerate(metrics_display):
        with metric_cols[mi]:
            st.markdown(f'<div style="font-size:0.8rem;color:#5B6B7D;font-weight:600;margin-bottom:6px;">{label.upper()}</div>')
            for mk in models_to_show:
                m = MODEL_METRICS[mk]
                val = m[key] * 100
                st.markdown(
                    f'<div class="metric-card" style="border-left:3px solid {m["color"]};margin-bottom:6px;padding:10px;">'
                    f'<div class="metric-num" style="color:{m["color"]};font-size:1.3rem;">{val:.1f}%</div>'
                    f'<div class="metric-label">{m["name"]}</div>'
                    f'</div>', unsafe_allow_html=True,
                )

    # --- Bootstrap CIs ---
    st.markdown("---")
    st.markdown("### \U0001F4CA Bootstrap 95% Confidence Intervals (Accuracy)")
    for mk in models_to_show:
        m = MODEL_METRICS[mk]
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E0E0E0;border-left:4px solid {m["color"]};'
            f'border-radius:8px;padding:10px 16px;margin-bottom:6px;">'
            f'<strong style="color:{m["color"]};">{m["name"]}</strong> \u2014 '
            f'Accuracy: <strong>{m["accuracy"]*100:.1f}%</strong> '
            f'<span style="color:#5B6B7D;">CI: {m["acc_ci"]}</span></div>',
            unsafe_allow_html=True,
        )

    # --- McNemar tests ---
    st.markdown("---")
    st.markdown("### \U0001F504 McNemar Tests (Statistical Significance)")
    for comparison, result in MCNEMAR_RESULTS.items():
        sig = "SIGNIFICANT" in result
        color = "#1E8449" if sig else "#E67E22"
        st.markdown(
            f'<div style="background:#FFFFFF;border:1px solid #E0E0E0;border-left:4px solid {color};'
            f'border-radius:8px;padding:10px 16px;margin-bottom:6px;">'
            f'<strong>{comparison}</strong> \u2014 {result}</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<p style="color:#5B6B7D;font-size:0.85rem;margin-top:10px;">'
        'Key finding: MobileNetV2 and Ensemble are <strong>statistically significantly better</strong> '
        'than both LR and CNN. The CNN alone is NOT significantly better than the LR baseline (p=0.249).</p>',
        unsafe_allow_html=True,
    )

    # --- Model architectures ---
    st.markdown("---")
    st.markdown("### \U0001F5BC Architectures")

    with st.expander("LR Baseline (Handcrafted Features)"):
        st.markdown("**Parameters:** 6 (5 feature coefficients + 1 intercept)")
        st.markdown("**Input:** 5 geometric features (entropy, smoothness, radius, intersections, displacement)")
        st.markdown("**Training:** Patient-level 5-fold StratifiedGroupKFold CV, class_weight='balanced'")
        st.markdown("**Top feature:** stroke_entropy (LR coefficient +1.53)")

    with st.expander("CNN v4 (3-Block, Oversampling)"):
        st.markdown("**Parameters:** ~110,000")
        st.markdown("**Architecture:**")
        st.markdown("- Input (256, 256, 1)")
        st.markdown("- Conv2D(32) \u2192 ReLU \u2192 MaxPool \u2192 Dropout(0.25)")
        st.markdown("- Conv2D(64) \u2192 ReLU \u2192 MaxPool \u2192 Dropout(0.25)")
        st.markdown("- Conv2D(128) \u2192 ReLU \u2192 MaxPool")
        st.markdown("- GlobalAveragePooling2D")
        st.markdown("- Dense(64) \u2192 ReLU \u2192 Dropout(0.4)")
        st.markdown("- Dense(1, Sigmoid)")
        st.markdown("**Training:** 5-fold CV + oversampling healthy to 50/50, lr=1e-4, 60 epochs, batch=16")
        st.markdown("**Collapse history:** v1-v3 collapsed (predicted all PD). v4 fixed via oversampling.")

    with st.expander("MobileNetV2 (Transfer Learning)"):
        st.markdown("**Parameters:** ~3.5M frozen (pretrained) + ~8K trainable")
        st.markdown("**Architecture:**")
        st.markdown("- Input (256, 256, 3) \u2014 grayscale replicated to 3 channels")
        st.markdown("- MobileNetV2 base (pretrained on ImageNet, frozen)")
        st.markdown("- GlobalAveragePooling2D")
        st.markdown("- Dense(64) \u2192 ReLU \u2192 Dropout(0.4)")
        st.markdown("- Dense(1, Sigmoid)")
        st.markdown("**Training:** 5-fold CV + oversampling, lr=1e-4, 40 epochs, batch=16")
        st.markdown("**Why it works:** Pretrained features from ImageNet transfer well to spiral images \u2014 edges, curves, textures are general visual patterns.")

    with st.expander("Ensemble (3-Way Average)"):
        st.markdown("**Method:** Average the output probabilities of LR + CNN + MobileNetV2")
        st.markdown("**Why it works:** The 3 models make different errors. Averaging cancels out individual mistakes.")
        st.markdown("**Result:** 92.9% accuracy, 98.8% recall \u2014 the best model overall")

    # --- Confusion matrices ---
    st.markdown("---")
    st.markdown("### \U0001F4CA Confusion Matrices (Pooled OOF)")

    cm_cols = st.columns(4)
    cm_data = {
        "lr": ("LR Baseline", "TN/FP/FN/TP from training"),
        "cnn": ("CNN v4", "131 / 81 / 26 / 394"),
        "mobilenet": ("MobileNetV2", "172 / 40 / 27 / 393"),
        "ensemble": ("Ensemble", "172 / 40 / 5 / 415"),
    }

    for i, (mk, (name, cm_str)) in enumerate(cm_data.items()):
        with cm_cols[i]:
            m = MODEL_METRICS[mk]
            st.markdown(f'**{name}**')
            st.markdown(
                f'<div class="metric-card" style="border-top:3px solid {m["color"]};font-family:monospace;font-size:0.85rem;">'
                f'<div style="color:#5B6B7D;margin-bottom:4px;">{cm_str}</div>'
                f'<div style="color:{m["color"]};font-weight:700;">Acc: {m["accuracy"]*100:.1f}%</div>'
                f'<div style="color:#5B6B7D;">Rec: {m["recall"]*100:.1f}%</div>'
                f'</div>', unsafe_allow_html=True,
            )

    # --- Summary ---
    st.markdown("---")
    st.markdown("### \U0001F4DD Summary")
    st.markdown(
        '<div style="background:#EDF3F8;border-radius:10px;padding:16px;font-size:0.9rem;color:#1A2B3C;">'
        '<strong>Key findings:</strong><br>'
        '1. The handcrafted LR baseline (78.8% accuracy) is an honest, interpretable model.<br>'
        '2. The CNN (81.5%) is slightly better but NOT statistically significant (p=0.249) \u2014 '
        'training from scratch on 632 images is insufficient.<br>'
        '3. MobileNetV2 (89.4%) is SIGNIFICANTLY better than both (p<0.001) \u2014 transfer learning helps on small data.<br>'
        '4. The ensemble (92.9% accuracy, 98.8% recall) is the best model, SIGNIFICANTLY better than all individual models.<br><br>'
        '<strong>Recommended model for deployment:</strong> Ensemble \u2014 highest accuracy and recall, '
        'catches 98.8% of PD cases while keeping false alarms manageable.'
        '</div>',
        unsafe_allow_html=True,
    )


st.markdown("---")
st.caption(
    "\U0001F9E0 Parkinson\u2019s Spiral Drawing Screening Aid \u00b7 4-Model Comparison \u00b7 "
    "Educational demo, not a clinical diagnostic tool."
)
