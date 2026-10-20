"""
Streamlit demo for Parkinson's Spiral Drawing Screening Aid.

Design philosophy:
  - Lean CSS, work WITH Streamlit (not against it)
  - Native st.columns() for bento grid (no custom CSS grid)
  - Native st.toggle() for theme switch (no custom HTML pills)
  - Minimal sidebar styling — let Streamlit's native collapse/expand work
  - Neomorphism accents via small CSS utility classes, not full overrides
"""

import os
import sys
import base64
from io import BytesIO
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
# Feature info (plain language explanations)
# =============================================================================
FEATURE_INFO = {
    "radius_deviation": {
        "title": "Spiral Shape Irregularity",
        "plain": "How far the spiral deviates from a perfect Archimedean spiral.",
        "direction": "higher",
        "cue": "Higher = more irregular",
        "decimals": 3,
    },
    "stroke_smoothness": {
        "title": "Stroke Jerkiness",
        "plain": "How jerky the hand movement was — sudden direction changes.",
        "direction": "higher",
        "cue": "Higher = jerkier",
        "decimals": 4,
    },
    "stroke_entropy": {
        "title": "Direction Randomness",
        "plain": "How random the stroke directions are — PD strokes are more chaotic.",
        "direction": "higher",
        "cue": "Higher = more random",
        "decimals": 3,
    },
    "intersection_count": {
        "title": "Self-Intersections",
        "plain": "How many times the spiral crosses itself.",
        "direction": "higher",
        "cue": "Higher = more crossings",
        "decimals": 0,
    },
    "mean_squared_displacement": {
        "title": "Spiral Spread",
        "plain": "How spread out the spiral is. PD spirals are smaller (micrographia).",
        "direction": "lower",
        "cue": "Lower = smaller spiral",
        "decimals": 4,
    },
}

FEATURE_ORDER = [
    "stroke_entropy",
    "stroke_smoothness",
    "intersection_count",
    "mean_squared_displacement",
    "radius_deviation",
]

# =============================================================================
# Model metrics (verified local evaluate.py run, 5-fold patient-level CV)
# =============================================================================
MODEL_METRICS = {
    "lr":        {"name": "LR Baseline",  "accuracy": 0.7880, "recall": 0.8024, "precision": 0.8686, "f1": 0.8342,
                  "acc_ci": "[0.7547, 0.8196]", "params": "6 parameters", "color": "#1E6B3A", "auc": 0.8539},
    "cnn":       {"name": "CNN v4",       "accuracy": 0.8149, "recall": 0.9786, "precision": 0.7919, "f1": 0.8754,
                  "acc_ci": "[0.7832, 0.8434]", "params": "~110K parameters", "color": "#1F618D", "auc": 0.8580},
    "mobilenet": {"name": "MobileNetV2",  "accuracy": 0.8940, "recall": 0.9357, "precision": 0.9076, "f1": 0.9215,
                  "acc_ci": "[0.8703, 0.9177]", "params": "~3.5M frozen + 8K trainable", "color": "#B9770E", "auc": 0.9590},
    "ensemble":  {"name": "Ensemble",     "accuracy": 0.9446, "recall": 0.9929, "precision": 0.9287, "f1": 0.9597,
                  "acc_ci": "[0.9256, 0.9620]", "params": "Average of LR + CNN + MobileNetV2", "color": "#6C3483", "auc": 0.9852},
}

MCNEMAR_RESULTS = {
    "CNN v4 vs LR Baseline":         "p = 0.249 (not significant)",
    "MobileNetV2 vs LR Baseline":    "p < 0.001 (significant)",
    "MobileNetV2 vs CNN v4":         "p < 0.001 (significant)",
    "Ensemble vs LR Baseline":       "p < 0.001 (significant)",
    "Ensemble vs CNN v4":            "p < 0.001 (significant)",
    "Ensemble vs MobileNetV2":       "p < 0.001 (significant)",
}

CM_DATA = {
    "lr":        ("LR Baseline",  (161, 51, 83, 337)),
    "cnn":       ("CNN v4",       (104, 108, 9, 411)),
    "mobilenet": ("MobileNetV2",  (172, 40, 27, 393)),
    "ensemble":  ("Ensemble",     (180, 32, 3, 417)),
}

# Brain SVG icon (inline, no emoji)
BRAIN_SVG = '''
<svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
  <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
  <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>
</svg>
'''


# =============================================================================
# Theme state — must be set BEFORE any CSS is injected
# =============================================================================
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"


# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Parkinson's Spiral Screening",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CSS — minimal, theme-aware, neomorphism accents
# =============================================================================
def get_css(theme):
    """Return CSS for the given theme. Variables drive everything.

    Design priorities:
    - Large fonts (18px+ body) for elderly users on Screening Demo
    - High contrast in dark mode (no dim secondary text)
    - Projector-friendly on Model Details (big section headers, big tables)
    """
    if theme == "dark":
        vars_css = """
        :root {
            --bg: #181B21;
            --surface: #181B21;
            --text: #FFFFFF;
            --text-secondary: #E5E7EB;
            --text-tertiary: #C9CED6;
            --accent: #2DD4BF;
            --accent-bright: #5EEAD4;
            --accent-dim: #134E4A;
            --pos: #F87171;
            --pos-bright: #EF4444;
            --neg: #4ADE80;
            --neg-bright: #10B981;
            --shadow-dark: #0A0C10;
            --shadow-light: #262A33;
            --border: #3A3F4A;
            --badge-bg: #134E4A;
            --badge-text: #5EEAD4;
            --warning: #FBBF24;
            --warning-bg: rgba(251, 191, 36, 0.12);
        }
        """
    else:
        vars_css = """
        :root {
            --bg: #F4F6F8;
            --surface: #F4F6F8;
            --text: #0F172A;
            --text-secondary: #334155;
            --text-tertiary: #475569;
            --accent: #0E7C7B;
            --accent-bright: #0E7C7B;
            --accent-dim: #B2DDDC;
            --pos: #B91C1C;
            --pos-bright: #DC2626;
            --neg: #047857;
            --neg-bright: #059669;
            --shadow-dark: #CBD5E1;
            --shadow-light: #FFFFFF;
            --border: #CBD5E1;
            --badge-bg: #E0F2F1;
            --badge-text: #0E7C7B;
            --warning: #B45309;
            --warning-bg: rgba(217, 119, 6, 0.08);
        }
        """
    return vars_css + """

    /* ---------- Base typography (LARGER for readability) ---------- */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", sans-serif;
        color: var(--text);
        font-size: 17px;
        line-height: 1.6;
    }
    h1, h2, h3, h4 { color: var(--text); font-weight: 700; letter-spacing: -0.01em; }

    /* ---------- Streamlit background ---------- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: var(--bg);
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        background-color: var(--bg);
    }

    /* ---------- Layout: responsive to sidebar state ---------- */
    /* When sidebar is OPEN (default), main content has room */
    /* When sidebar is COLLAPSED, main content expands to fill */
    /* Streamlit handles the actual flex layout - we just remove width caps */
    .block-container {
        max-width: 100% !important;
        padding-top: 24px !important;
        padding-left: 24px !important;
        padding-right: 24px !important;
        width: 100% !important;
    }
    [data-testid="stAppViewContainer"] > section > div {
        max-width: 100% !important;
    }
    /* Make sure the main section grows to fill available space */
    [data-testid="stAppViewContainer"] > section {
        flex: 1 1 auto !important;
        min-width: 0 !important;
    }
    /* Sidebar transition should be smooth */
    [data-testid="stSidebar"] {
        transition: width 0.3s ease, transform 0.3s ease !important;
    }

    /* ---------- Columns: ensure they fill their container ---------- */
    [data-testid="stHorizontalBlock"] {
        gap: 16px;
        align-items: stretch !important;
        width: 100% !important;
    }
    [data-testid="stHorizontalBlock"] > div {
        flex: 1 1 0 !important;
        min-width: 0 !important;
    }

    /* ---------- Bento cards (neomorphism accents, LARGER padding) ---------- */
    .bento {
        background: var(--surface);
        border-radius: 16px;
        padding: 24px 26px;
        box-shadow: 6px 6px 12px var(--shadow-dark), -6px -6px 12px var(--shadow-light);
        margin-bottom: 16px;
        transition: box-shadow 0.2s ease, transform 0.2s ease;
    }
    .bento:hover {
        box-shadow: 8px 8px 16px var(--shadow-dark), -8px -8px 16px var(--shadow-light);
        transform: translateY(-1px);
    }
    .bento.featured {
        box-shadow: 6px 6px 12px var(--shadow-dark), -6px -6px 12px var(--shadow-light),
                    0 0 0 3px var(--accent);
        padding: 28px 30px;
    }
    .bento.compact {
        padding: 18px 20px;
        border-radius: 14px;
    }

    /* ---------- Hero header (BIGGER for impact) ---------- */
    .hero {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 24px 28px;
        margin-bottom: 20px;
    }
    .hero-icon {
        flex: 0 0 auto;
        width: 64px;
        height: 64px;
        border-radius: 16px;
        background: var(--surface);
        box-shadow: inset 3px 3px 6px var(--shadow-dark), inset -3px -3px 6px var(--shadow-light);
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--accent);
    }
    .hero-icon svg { width: 36px; height: 36px; }
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text);
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: var(--text-secondary);
        margin: 6px 0 0 0;
    }

    /* ---------- Section header (BIG and BOLD for projector readability) ---------- */
    .section-h {
        font-size: 1.35rem;
        font-weight: 700;
        color: var(--accent);
        margin: 32px 0 16px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid var(--accent);
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .section-h::before {
        content: "";
        display: inline-block;
        width: 6px;
        height: 24px;
        background: var(--accent);
        border-radius: 3px;
    }

    /* ---------- Prediction display (LARGER for elderly users) ---------- */
    .model-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: var(--text);
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 8px;
    }
    .model-dot {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    .hero-prob {
        font-family: "SF Mono", "Menlo", monospace;
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 8px 0;
    }
    .hero-prob.pos { color: var(--pos-bright); }
    .hero-prob.neg { color: var(--neg-bright); }
    .compact-prob {
        font-family: "SF Mono", "Menlo", monospace;
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 6px 0;
    }
    .compact-prob.pos { color: var(--pos-bright); }
    .compact-prob.neg { color: var(--neg-bright); }
    .pred-label {
        display: inline-block;
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 6px 14px;
        border-radius: 6px;
        margin-top: 8px;
        color: white !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    }
    .pred-label.pos { background: var(--pos-bright); }
    .pred-label.neg { background: var(--neg-bright); }

    /* ---------- Progress bar (THICKER for visibility) ---------- */
    .prob-track {
        height: 10px;
        background: var(--surface);
        border-radius: 5px;
        box-shadow: inset 2px 2px 4px var(--shadow-dark), inset -2px -2px 4px var(--shadow-light);
        margin-top: 14px;
        overflow: hidden;
    }
    .prob-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.3s ease;
    }

    /* ---------- Recommended badge (BIGGER) ---------- */
    .badge {
        display: inline-block;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 5px 12px;
        border-radius: 6px;
        background: var(--badge-bg);
        color: var(--badge-text);
        margin-left: auto;
    }

    /* ---------- Disclaimer (BIGGER, CLEARER) ---------- */
    .disclaimer {
        background: var(--warning-bg);
        border-left: 4px solid var(--warning);
        padding: 14px 18px;
        border-radius: 6px;
        font-size: 1rem;
        color: var(--text);
        margin-bottom: 20px;
        line-height: 1.5;
    }
    .disclaimer strong {
        color: var(--warning);
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        display: block;
        margin-bottom: 6px;
        font-weight: 800;
    }

    /* ---------- Feature card (LARGER for elderly) ---------- */
    .feat-name {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1.3;
    }
    .feat-desc {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin: 6px 0 10px 0;
        line-height: 1.5;
    }
    .feat-val {
        font-family: "SF Mono", "Menlo", monospace;
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text);
    }
    .feat-cue {
        font-size: 0.85rem;
        color: var(--text-secondary);
        margin-top: 6px;
        font-weight: 600;
    }

    /* ---------- Metric table (BIGGER for projector) ---------- */
    table.metrics {
        width: 100%;
        border-collapse: collapse;
        font-size: 1.05rem;
    }
    table.metrics th {
        text-align: left;
        font-weight: 700;
        color: var(--text);
        padding: 14px 16px;
        border-bottom: 3px solid var(--accent);
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        background: var(--surface);
    }
    table.metrics td {
        padding: 16px;
        border-bottom: 1px solid var(--border);
        color: var(--text);
    }
    table.metrics td.num {
        font-family: "SF Mono", "Menlo", monospace;
        text-align: right;
        font-weight: 600;
    }
    table.metrics tr.best td {
        background: rgba(45, 212, 191, 0.12);
        font-weight: 700;
    }
    table.metrics tr.best td:first-child {
        border-left: 4px solid var(--accent);
    }
    .color-chip {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 3px;
        margin-right: 10px;
        vertical-align: middle;
    }

    /* ---------- Confusion matrix (BIGGER for projector) ---------- */
    .cm-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 8px;
        margin-top: 14px;
    }
    .cm-cell {
        padding: 14px 10px;
        border-radius: 8px;
        text-align: center;
        background: var(--surface);
        box-shadow: inset 2px 2px 4px var(--shadow-dark), inset -2px -2px 4px var(--shadow-light);
    }
    .cm-cell .lbl {
        font-size: 0.85rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .cm-cell .val {
        font-family: "SF Mono", "Menlo", monospace;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 6px;
    }
    .cm-cell.tn .val, .cm-cell.tp .val { color: var(--neg-bright); }
    .cm-cell.fp .val, .cm-cell.fn .val { color: var(--pos-bright); }

    /* ---------- Status dot (BIGGER) ---------- */
    .status-row {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1rem;
        color: var(--text);
        padding: 6px 0;
    }
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .status-dot.ok { background: var(--neg-bright); }
    .status-dot.no { background: var(--pos-bright); }

    /* ---------- Sidebar section ---------- */
    .sb-section {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--accent);
        margin: 20px 0 10px 0;
    }

    /* ---------- Streamlit native tweaks (BIGGER buttons) ---------- */
    .stButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px 24px !important;
        min-height: 48px;
    }
    .stButton > button:hover {
        background: var(--accent-dim) !important;
        color: white !important;
    }

    /* ---------- Bigger sidebar widget labels ---------- */
    [data-testid="stSidebar"] label {
        font-size: 1rem !important;
        color: var(--text) !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stRadio > div {
        font-size: 1rem !important;
    }

    /* ---------- FORCE Streamlit native widgets to inherit theme text color ---------- */
    /* This fixes the dark-text-on-dark-bg issue in expanders, markdown, file uploader, etc. */
    .stApp, [data-testid="stAppViewContainer"],
    .stApp p, .stApp span, .stApp li, .stApp div, .stApp label,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .streamlit-expanderHeader,
    .streamlit-expanderContent,
    .streamlit-expanderContent p,
    .streamlit-expanderContent li,
    .streamlit-expanderContent span,
    .streamlit-expanderContent strong,
    .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown strong, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5,
    .stText, .stText p,
    .stFileUploader, .stFileUploader p, .stFileUploader span, .stFileUploader label,
    .stRadio, .stRadio label, .stRadio span,
    .stToggle, .stToggle label, .stToggle span,
    .stCheckbox, .stCheckbox label, .stCheckbox span,
    .stButton, .stButton label,
    details, summary {
        color: var(--text) !important;
    }
    /* Links and code blocks keep their natural colors */
    .stApp a { color: var(--accent); }
    .stApp code {
        background: var(--surface);
        color: var(--text);
        padding: 2px 6px;
        border-radius: 4px;
        box-shadow: inset 1px 1px 2px var(--shadow-dark), inset -1px -1px 2px var(--shadow-light);
    }

    /* ---------- File uploader styling (theme-aware) ---------- */
    [data-testid="stFileUploader"] {
        background: var(--surface) !important;
        border-radius: 12px !important;
        padding: 12px !important;
        box-shadow: inset 2px 2px 4px var(--shadow-dark), inset -2px -2px 4px var(--shadow-light) !important;
    }
    [data-testid="stFileUploader"] section {
        background: var(--surface) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: var(--surface) !important;
        color: var(--text) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--text) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--accent) !important;
    }

    /* ---------- Radio, toggle, checkbox buttons ---------- */
    .stRadio > div, .stToggle, .stCheckbox {
        color: var(--text) !important;
    }
    .stRadio label, .stToggle label, .stCheckbox label {
        color: var(--text) !important;
    }

    /* ---------- Section heading (the ##### markdown) - make it pop ---------- */
    .stApp h5 {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        color: var(--accent) !important;
        margin-top: 16px !important;
        margin-bottom: 12px !important;
    }
    .stApp h6 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: var(--text) !important;
    }

    /* ---------- Expander header should be bold and clearly visible ---------- */
    .streamlit-expanderHeader {
        font-size: 1.1rem !important;
        font-weight: 700 !important;
        color: var(--text) !important;
        background: var(--surface) !important;
        border-radius: 12px !important;
        padding: 12px 16px !important;
    }
    /* Expander container should match the bento card style */
    details {
        background: var(--surface) !important;
        border: none !important;
        border-radius: 14px !important;
        box-shadow: 4px 4px 8px var(--shadow-dark), -4px -4px 8px var(--shadow-light) !important;
        margin-bottom: 12px !important;
        padding: 4px !important;
    }
    .streamlit-expanderContent {
        padding: 16px 20px !important;
        font-size: 1rem !important;
        line-height: 1.7 !important;
        color: var(--text) !important;
    }
    .streamlit-expanderContent p,
    .streamlit-expanderContent li,
    .streamlit-expanderContent strong {
        color: var(--text) !important;
    }

    /* ---------- (column gaps already defined above) ---------- */

    /* ---------- (expander styles already defined above) ---------- */

    /* ---------- Hide streamlit chrome ---------- */
    #MainMenu, footer { visibility: hidden; }
    """


st.markdown(f"<style>{get_css(st.session_state['theme'])}</style>", unsafe_allow_html=True)


# =============================================================================
# Model loaders
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
    model = tf.keras.Model(model.inputs, model.outputs, name=model.name)
    weights_path = "data/processed/cnn_model_weights.weights.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        return model, True
    return None, False


@st.cache_resource
def load_mobilenet_model():
    import tensorflow as tf
    import warnings
    warnings.filterwarnings("ignore", message=".*input_shape.*")
    base = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet",
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(256, 256, 3), name="input_256")
    x = tf.keras.layers.Resizing(224, 224, name="resize_to_224")(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs, name="spiral_mobilenetv2")
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
# Sidebar — works WITH Streamlit (native toggle, native collapse)
# =============================================================================
st.sidebar.markdown('<div class="sb-section">Theme</div>', unsafe_allow_html=True)
is_dark = st.sidebar.toggle("Dark mode", value=(st.session_state["theme"] == "dark"))
st.session_state["theme"] = "dark" if is_dark else "light"

st.sidebar.markdown('<div class="sb-section">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Page", ["Screening Demo", "Model Details"], label_visibility="collapsed")

st.sidebar.markdown('<div class="sb-section">Model Status</div>', unsafe_allow_html=True)
def status_row(loaded, name):
    cls = "ok" if loaded else "no"
    label = "Loaded" if loaded else "Not loaded"
    st.sidebar.markdown(
        f'<div class="status-row"><span class="status-dot {cls}"></span>{name} — {label}</div>',
        unsafe_allow_html=True,
    )
status_row(True, "LR Baseline")
status_row(cnn_available, "CNN v4")
status_row(mobilenet_available, "MobileNetV2")

st.sidebar.markdown('<div class="sb-section">Dataset</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div style="font-size:0.82rem;color:var(--text-secondary);line-height:1.6;">'
    '632 images · 158 patients<br>420 PD · 212 Healthy</div>',
    unsafe_allow_html=True,
)


# =============================================================================
# Helper: matplotlib figure to base64 (so we can embed in HTML for tight layout)
# =============================================================================
def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


# =============================================================================
# PAGE 1: Screening Demo
# =============================================================================
if page == "Screening Demo":

    # Hero header
    st.markdown(
        f'<div class="bento hero">'
        f'<div class="hero-icon">{BRAIN_SVG}</div>'
        f'<div><div class="hero-title">Parkinson\'s Spiral Drawing Screening Aid</div>'
        f'<div class="hero-subtitle">Upload a spiral drawing to screen for Parkinson\'s disease indicators</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="disclaimer"><strong>For research only · Not for clinical use</strong>'
        'This is an educational screening tool, not a medical device. '
        'It cannot diagnose Parkinson\'s disease. Any positive result should be discussed with a doctor.</div>',
        unsafe_allow_html=True,
    )

    # ---------- INPUT CARD: in the main page (not sidebar) ----------
    st.markdown('<div class="section-h">Choose a Spiral Drawing</div>', unsafe_allow_html=True)

    image_to_test = None
    true_label = None
    true_filename = None

    # CSS: target the horizontal block that contains the file uploader.
    # :has() is supported in all modern browsers (Chrome 105+, Safari 15.4+, Firefox 121+).
    # Both columns get equal height + flex centering, so the random button card
    # matches the file uploader card height regardless of content.
    st.markdown(
        '<style>'
        '[data-testid="stHorizontalBlock"]:has([data-testid="stFileUploader"]) > div '
        '{ min-height: 200px; display: flex; flex-direction: column; justify-content: center; }'
        '</style>',
        unsafe_allow_html=True,
    )

    in_col1, in_col2 = st.columns([1, 1])

    with in_col1:
        st.markdown("##### Upload your own spiral image")
        uploaded_file = st.file_uploader("Choose a spiral image (PNG or JPG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                img = img.astype(np.float32) / 255.0
                img = np.expand_dims(img, axis=-1)
                image_to_test = img
                true_filename = uploaded_file.name

    with in_col2:
        st.markdown("##### Or try a random sample from our dataset")
        if st.button("Show me a random spiral", key="random_btn", use_container_width=True):
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
        # Caption under the button to balance height with the file uploader's hint text
        st.caption("Picks any of the 632 spiral drawings from our dataset")

    if image_to_test is not None:
        # Run all models
        with st.spinner("Analyzing the spiral drawing..."):
            features_dict = extract_all_features(image_to_test)
            fv = np.array([features_dict[n] for n in feature_names]).reshape(1, -1)
            fv_s = scaler.transform(fv)
            lr_prob = float(lr_model.predict_proba(fv_s)[0, 1])
            lr_pred = 1 if lr_prob >= 0.5 else 0

            cnn_prob = None
            if cnn_available and cnn_model is not None:
                img_batch = np.expand_dims(image_to_test, axis=0)
                cnn_prob = float(cnn_model.predict(img_batch, verbose=0)[0, 0])
                cnn_pred = 1 if cnn_prob >= 0.5 else 0

            mb_prob = None
            if mobilenet_available and mobilenet_model is not None:
                img_3ch = np.repeat(np.expand_dims(image_to_test, axis=0), 3, axis=-1)
                mb_prob = float(mobilenet_model.predict(img_3ch, verbose=0)[0, 0])
                mb_pred = 1 if mb_prob >= 0.5 else 0

            all_probs = [lr_prob]
            if cnn_prob is not None: all_probs.append(cnn_prob)
            if mb_prob is not None: all_probs.append(mb_prob)
            ens_prob = float(np.mean(all_probs)) if len(all_probs) > 1 else None
            ens_pred = 1 if (ens_prob is not None and ens_prob >= 0.5) else (0 if ens_prob is not None else None)

        # ---------- BENTO ROW 1: image (4) + ensemble featured (8) ----------
        st.markdown('<div class="section-h">Spiral Drawing & Final Verdict</div>', unsafe_allow_html=True)
        col_img, col_ens = st.columns([4, 8])

        with col_img:
            # Smaller image - 3.2" instead of 3.8" - fits better in the narrower column
            fig, ax = plt.subplots(figsize=(3.2, 3.2), constrained_layout=True)
            ax.imshow(image_to_test.squeeze(), cmap="gray")
            ax.axis("off")
            if true_filename:
                ax.set_title(true_filename, fontsize=8, color="#9CA3AF", pad=4)
            img_b64 = fig_to_base64(fig)
            plt.close(fig)

            true_label_html = ""
            if true_label is not None:
                ls = "Parkinson's" if true_label == 1 else "Healthy"
                lc = "var(--pos-bright)" if true_label == 1 else "var(--neg-bright)"
                true_label_html = (
                    f'<div style="margin-top:12px;font-size:0.9rem;color:var(--text-secondary);'
                    f'text-align:center;padding:8px;border-radius:6px;'
                    f'background:var(--surface);box-shadow:inset 1px 1px 2px var(--shadow-dark),'
                    f'inset -1px -1px 2px var(--shadow-light);">'
                    f'True label: <strong style="color:{lc};font-size:1rem;">{ls}</strong></div>'
                )

            st.markdown(
                f'<div class="bento" style="text-align:center;">'
                f'<img src="data:image/png;base64,{img_b64}" style="width:100%;max-width:280px;border-radius:8px;display:block;margin:0 auto;">'
                f'{true_label_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

        with col_ens:
            if ens_prob is not None:
                cls = "pos" if ens_pred == 1 else "neg"
                label = "PD-positive" if ens_pred == 1 else "Healthy"
                banner_label = "PARKINSON'S DETECTED" if ens_pred == 1 else "NO PARKINSON'S DETECTED"
                pct = ens_prob * 100
                accent = MODEL_METRICS["ensemble"]["color"]

                # Big status banner at the bottom to fill the vertical space
                banner_bg = "var(--pos-bright)" if ens_pred == 1 else "var(--neg-bright)"
                banner_html = (
                    f'<div style="margin-top:20px;padding:18px 24px;border-radius:10px;'
                    f'background:{banner_bg};color:white;text-align:center;'
                    f'box-shadow:0 4px 12px rgba(0,0,0,0.15);">'
                    f'<div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;'
                    f'letter-spacing:0.1em;opacity:0.9;margin-bottom:4px;">Final Verdict</div>'
                    f'<div style="font-size:1.5rem;font-weight:800;letter-spacing:0.02em;">{banner_label}</div>'
                    f'<div style="font-size:0.9rem;margin-top:6px;opacity:0.95;">'
                    f'Based on the average of 3 models · {pct:.1f}% confidence</div>'
                    f'</div>'
                )

                st.markdown(
                    f'<div class="bento featured" style="height:100%;">'
                    f'<div class="model-name"><span class="model-dot" style="background:{accent};"></span>Ensemble<span class="badge">Recommended</span></div>'
                    f'<div class="hero-prob {cls}">{pct:.1f}%</div>'
                    f'<div class="pred-label {cls}">{label}</div>'
                    f'<div style="font-size:0.95rem;color:var(--text-secondary);margin-top:12px;line-height:1.5;">Average of LR + CNN + MobileNetV2 probabilities</div>'
                    f'<div class="prob-track"><div class="prob-fill" style="background:{accent};width:{pct:.1f}%;"></div></div>'
                    f'{banner_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # ---------- BENTO ROW 2: 3 compact model cards + summary ----------
        st.markdown('<div class="section-h">Individual Model Results</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)

        def compact_card(col, key, pred, prob, subtitle):
            with col:
                if prob is None:
                    st.markdown(
                        f'<div class="bento compact">'
                        f'<div class="model-name"><span class="model-dot" style="background:var(--text-tertiary);"></span>{MODEL_METRICS[key]["name"]}</div>'
                        f'<div style="font-size:1rem;color:var(--text-secondary);padding:10px 0;font-weight:600;">Not loaded</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    return
                cls = "pos" if pred == 1 else "neg"
                label = "POSITIVE" if pred == 1 else "NEGATIVE"
                pct = prob * 100
                accent = MODEL_METRICS[key]["color"]
                st.markdown(
                    f'<div class="bento compact">'
                    f'<div class="model-name"><span class="model-dot" style="background:{accent};"></span>{MODEL_METRICS[key]["name"]}</div>'
                    f'<div class="compact-prob {cls}">{pct:.1f}%</div>'
                    f'<div class="pred-label {cls}">{label}</div>'
                    f'<div style="font-size:0.85rem;color:var(--text-secondary);margin-top:10px;line-height:1.4;">{subtitle}</div>'
                    f'<div class="prob-track"><div class="prob-fill" style="background:{accent};width:{pct:.1f}%;"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        compact_card(c1, "lr", lr_pred, lr_prob, "Handcrafted geometric features")
        compact_card(c2, "cnn", cnn_pred if cnn_prob is not None else None, cnn_prob, "Neural network, trained from scratch")
        compact_card(c3, "mobilenet", mb_pred if mb_prob is not None else None, mb_prob, "Transfer learning, ImageNet pretrained")

        with c4:
            # Summary card: agreement
            preds_list = [lr_pred]
            if cnn_prob is not None: preds_list.append(cnn_pred)
            if mb_prob is not None: preds_list.append(mb_pred)
            if ens_prob is not None: preds_list.append(ens_pred)
            agree = len(set(preds_list)) == 1
            if agree:
                msg = "All models agree" if preds_list[0] == 0 else "All models: PD-positive"
                color = "var(--neg)" if preds_list[0] == 0 else "var(--pos)"
            else:
                pd_count = sum(preds_list)
                hl_count = len(preds_list) - pd_count
                msg = f"Models split: {pd_count} pos / {hl_count} neg"
                color = "#D97706"
            st.markdown(
                f'<div class="bento compact">'
                f'<div class="model-name">Agreement</div>'
                f'<div style="font-size:0.95rem;font-weight:600;color:{color};margin-top:6px;line-height:1.4;">{msg}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # ---------- BENTO ROW 3: 5 geometric features ----------
        st.markdown('<div class="section-h">Geometric Features (Used by LR Baseline)</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:1rem;color:var(--text-secondary);margin-top:-8px;margin-bottom:16px;line-height:1.5;">'
            'These five measurements describe the shape of the spiral. The LR baseline uses them; '
            'CNN and MobileNetV2 read raw pixels instead.</div>',
            unsafe_allow_html=True,
        )

        def pct_rank(value, name):
            if REF_STATS is None or name not in REF_STATS:
                return None
            s = REF_STATS[name]
            if s["max"] == s["min"]:
                return 50.0
            return ((value - s["min"]) / (s["max"] - s["min"])) * 100.0

        feat_cols = st.columns(5)
        for i, name in enumerate(FEATURE_ORDER):
            if name not in features_dict:
                continue
            info = FEATURE_INFO[name]
            value = features_dict[name]
            if info["decimals"] == 0:
                vs = f"{int(round(value))}"
            else:
                vs = f"{value:.{info['decimals']}f}"
            rank = pct_rank(value, name)
            rank_html = ""
            if rank is not None:
                rank_html = f'<div class="prob-track" style="margin-top:8px;height:4px;"><div class="prob-fill" style="background:var(--accent);width:{rank:.0f}%;"></div></div>'
            with feat_cols[i]:
                st.markdown(
                    f'<div class="bento compact">'
                    f'<div class="feat-name">{info["title"]}</div>'
                    f'<div class="feat-desc">{info["plain"]}</div>'
                    f'<div class="feat-val">{vs}</div>'
                    f'<div class="feat-cue">{info["cue"]}</div>'
                    f'{rank_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    else:
        # Empty state
        st.markdown(
            f'<div class="bento" style="padding:60px;text-align:center;margin-top:30px;">'
            f'<div class="hero-icon" style="margin:0 auto 16px auto;width:56px;height:56px;">{BRAIN_SVG}</div>'
            f'<div style="font-size:0.95rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;font-weight:700;">No image selected yet</div>'
            f'<div style="font-size:1.25rem;color:var(--text);font-weight:700;margin-bottom:8px;">Upload a spiral drawing above to begin</div>'
            f'<div style="color:var(--text-secondary);font-size:1rem;line-height:1.5;">Use the upload box or the random sample button at the top of this page.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# PAGE 2: Model Details
# =============================================================================
elif page == "Model Details":

    st.markdown(
        f'<div class="bento hero">'
        f'<div class="hero-icon">{BRAIN_SVG}</div>'
        f'<div><div class="hero-title">Model Details</div>'
        f'<div class="hero-subtitle">Architectures · metrics · statistical tests · patient-level 5-fold CV</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ---------- Performance table ----------
    st.markdown('<div class="section-h">Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:1rem;color:var(--text-secondary);margin-top:-8px;margin-bottom:16px;line-height:1.5;">'
        'Pooled out-of-fold predictions across 5 patient-level cross-validation folds. '
        '95% bootstrap confidence intervals in brackets (2000 resamples).</div>',
        unsafe_allow_html=True,
    )
    rows = ['<tr><th>Model</th><th>Accuracy</th><th>Recall</th><th>Precision</th><th>F1</th><th>AUC</th><th>95% CI (Acc)</th></tr>']
    for key in ["lr", "cnn", "mobilenet", "ensemble"]:
        m = MODEL_METRICS[key]
        is_best = (key == "ensemble")
        row_cls = ' class="best"' if is_best else ""
        rows.append(
            f'<tr{row_cls}>'
            f'<td><span class="color-chip" style="background:{m["color"]};"></span>{m["name"]}</td>'
            f'<td class="num">{m["accuracy"]*100:.2f}%</td>'
            f'<td class="num">{m["recall"]*100:.2f}%</td>'
            f'<td class="num">{m["precision"]*100:.2f}%</td>'
            f'<td class="num">{m["f1"]:.4f}</td>'
            f'<td class="num">{m["auc"]:.4f}</td>'
            f'<td class="num" style="color:var(--text-tertiary);">{m["acc_ci"]}</td>'
            f'</tr>'
        )
    st.markdown(
        f'<div class="bento"><table class="metrics">{"".join(rows)}</table></div>',
        unsafe_allow_html=True,
    )

    # ---------- Confusion matrices ----------
    st.markdown('<div class="section-h">Confusion Matrices (Pooled OOF)</div>', unsafe_allow_html=True)
    cm_cols = st.columns(4)
    for i, (mk, (name, (tn, fp, fn, tp))) in enumerate(CM_DATA.items()):
        m = MODEL_METRICS[mk]
        is_best = (mk == "ensemble")
        cls = " featured" if is_best else ""
        with cm_cols[i]:
            st.markdown(
                f'<div class="bento{cls}">'
                f'<div class="model-name"><span class="model-dot" style="background:{m["color"]};"></span>{name}</div>'
                f'<div class="cm-grid">'
                f'<div class="cm-cell tn"><div class="lbl">TN</div><div class="val">{tn}</div></div>'
                f'<div class="cm-cell fp"><div class="lbl">FP</div><div class="val">{fp}</div></div>'
                f'<div class="cm-cell fn"><div class="lbl">FN</div><div class="val">{fn}</div></div>'
                f'<div class="cm-cell tp"><div class="lbl">TP</div><div class="val">{tp}</div></div>'
                f'</div>'
                f'<div style="margin-top:10px;font-size:0.72rem;color:var(--text-tertiary);">Acc {m["accuracy"]*100:.1f}% · Recall {m["recall"]*100:.1f}%</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ---------- McNemar tests ----------
    st.markdown('<div class="section-h">Pairwise Significance (McNemar Test)</div>', unsafe_allow_html=True)
    rows = ['<tr><th>Comparison</th><th>p-value</th><th>Significant?</th></tr>']
    for comparison, result in MCNEMAR_RESULTS.items():
        sig = "significant" in result and "not" not in result
        sig_text = "Yes" if sig else "No"
        sig_color = "var(--neg)" if sig else "#D97706"
        rows.append(
            f'<tr><td>{comparison}</td>'
            f'<td class="num">{result}</td>'
            f'<td class="num" style="color:{sig_color};font-weight:600;">{sig_text}</td></tr>'
        )
    st.markdown(
        f'<div class="bento"><table class="metrics">{"".join(rows)}</table></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:1rem;color:var(--text-secondary);margin-top:12px;line-height:1.5;">'
        '<strong style="color:var(--text);">Key finding:</strong> MobileNetV2 and the Ensemble are statistically '
        'significantly better than both LR and CNN. CNN alone is <em>not</em> significantly better than the LR baseline.</div>',
        unsafe_allow_html=True,
    )

    # ---------- Architectures ----------
    st.markdown('<div class="section-h">Architectures</div>', unsafe_allow_html=True)
    with st.expander("LR Baseline (handcrafted features)"):
        st.markdown("**Parameters:** 6 (5 feature coefficients + 1 intercept)")
        st.markdown("**Input:** 5 geometric features")
        st.markdown("**Training:** Patient-level 5-fold StratifiedGroupKFold CV, class_weight='balanced'")
        st.markdown("**Strongest feature:** stroke_entropy (LR coefficient +1.53)")

    with st.expander("CNN v4 (3-block, trained from scratch)"):
        st.markdown("**Parameters:** ~110,000")
        st.markdown("- Conv2D(32) → ReLU → MaxPool → Dropout(0.25)")
        st.markdown("- Conv2D(64) → ReLU → MaxPool → Dropout(0.25)")
        st.markdown("- Conv2D(128) → ReLU → MaxPool")
        st.markdown("- GlobalAveragePooling2D → Dense(64) → Dropout(0.4) → Dense(1, Sigmoid)")
        st.markdown("**Training:** 5-fold CV + oversampling healthy to 50/50, lr=1e-4, 60 epochs, batch=16")
        st.markdown("**Design:** No BatchNorm, no L2. v1–v3 collapsed to 'predict all PD'; v4 fixed via oversampling.")

    with st.expander("MobileNetV2 (transfer learning)"):
        st.markdown("**Parameters:** ~3.5M frozen (ImageNet) + ~8K trainable (custom head)")
        st.markdown("- Input (256, 256, 3) — grayscale replicated to 3 channels")
        st.markdown("- MobileNetV2 base (pretrained, frozen)")
        st.markdown("- GlobalAveragePooling2D → Dense(64) → Dropout(0.4) → Dense(1, Sigmoid)")
        st.markdown("**Training:** 5-fold CV + oversampling, lr=1e-4, 40 epochs, batch=16")
        st.markdown("**Why it works:** Pretrained ImageNet features (edges, curves, textures) transfer well to spirals.")

    with st.expander("Ensemble (3-way probability average)"):
        st.markdown("**Method:** Average the output probabilities of LR + CNN + MobileNetV2")
        st.markdown("**Math:** P_ensemble = (P_LR + P_CNN + P_MobileNet) / 3, threshold at 0.5")
        st.markdown("**Why it works:** The three models make uncorrelated errors — averaging cancels the mistakes.")
        st.markdown("**Result:** 94.46% accuracy, 99.29% recall — significantly better than every individual model (p < 0.001).")

    # ---------- Summary ----------
    st.markdown('<div class="section-h">Summary</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="bento" style="font-size:1.05rem;line-height:1.8;">'
        '<strong style="color:var(--accent);font-size:1.15rem;">Key Findings</strong><br><br>'
        '<strong>1.</strong> LR baseline (78.80% accuracy) — honest, interpretable model.<br>'
        '<strong>2.</strong> CNN v4 (81.49%) — numerically better but NOT statistically significant vs LR (p = 0.249).<br>'
        '<strong>3.</strong> MobileNetV2 (89.40%) — significantly better than both (p < 0.001) — transfer learning helps on small data.<br>'
        '<strong>4.</strong> Ensemble (94.46% accuracy, 99.29% recall, AUC 0.985) — best model, significantly better than all individuals.<br><br>'
        '<strong style="color:var(--accent);font-size:1.15rem;">Recommended for deployment:</strong> Ensemble — catches 99.29% of PD cases (3 misses out of 420).'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Footer
# =============================================================================
st.markdown(
    '<div style="margin-top:32px;padding-top:16px;border-top:1px solid var(--border);'
    'font-size:0.72rem;color:var(--text-tertiary);">'
    'Parkinson\'s Spiral Drawing Screening Aid · MCA Mini-Project · '
    'Dataset: NewHandPD + HandPD (158 patients, 632 images) · '
    'Educational demo, not a clinical diagnostic tool.'
    '</div>',
    unsafe_allow_html=True,
)
