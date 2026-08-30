"""
Streamlit demo for Parkinson's Spiral Drawing Screening Aid (Handcrafted Baseline).

This app allows you to upload a spiral image or pick one from the dataset.
It extracts the 5 handcrafted features and uses a Logistic Regression
baseline to predict if the spiral is Healthy or Parkinson's-screening-positive.

Usage:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cv2

# Add project root to path so we can import from src/
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_newhandpd
from src.features import extract_all_features
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(
    page_title="Parkinson's Spiral Screening (Handcrafted)",
    page_icon="🌀",
    layout="wide"
)

st.title("Parkinson's Spiral Drawing Screening Aid")
st.markdown("**Handcrafted Features Baseline Demo**")
st.markdown("---")

# --- Cache the model loading and training so it only happens once ---
@st.cache_resource
def load_data_and_train_model():
    """Load dataset, features, and train the baseline LR model."""
    # Load dataset (for getting random samples)
    images, labels, patient_ids, filenames = load_newhandpd("data/raw/Merged")
    
    # Load pre-computed features
    features = np.load("data/processed/features_handcrafted.npy")
    feature_names = np.load("data/processed/feature_names.npy", allow_pickle=True).tolist()
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train Logistic Regression on ALL data (for demo purposes)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(features_scaled, labels)
    
    return images, labels, patient_ids, filenames, model, scaler, feature_names

try:
    images, labels, patient_ids, filenames, model, scaler, feature_names = load_data_and_train_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please run `python src/features.py` and `python src/train.py --model baseline` first.")
    st.stop()

# --- Sidebar: Input selection ---
st.sidebar.header("Choose Input Method")
input_method = st.sidebar.radio("Select an option:", ["Upload Image", "Pick Random Dataset Image"])

image_to_test = None
true_label = None
true_filename = None

if input_method == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload a spiral image (PNG/JPG)", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        # Read uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Preprocess: resize to 256x256 and normalize
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            image_to_test = img
            true_filename = uploaded_file.name
else:
    if st.sidebar.button("Pick a Random Image"):
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

# --- Main Content: Prediction and Visualization ---
if image_to_test is not None:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        # Display image
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(image_to_test.squeeze(), cmap="gray")
        ax.axis("off")
        if true_filename:
            ax.set_title(true_filename, fontsize=10)
        st.pyplot(fig)
        
        if true_label is not None:
            label_str = "Parkinson's" if true_label == 1 else "Healthy"
            st.info(f"**True Label:** {label_str}")
    
    with col2:
        st.subheader("Prediction (Handcrafted Baseline)")
        
        # Extract features for the test image
        with st.spinner("Extracting features..."):
            features_dict = extract_all_features(image_to_test)
            feature_vector = np.array([features_dict[name] for name in feature_names]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector)
        
        # Predict
        probability = model.predict_proba(feature_vector_scaled)[0, 1]
        prediction = 1 if probability >= 0.5 else 0
        prediction_str = "Parkinson's-screening-positive" if prediction == 1 else "Healthy"
        
        # Display result
        if prediction == 1:
            st.error(f"**Result: {prediction_str}**")
        else:
            st.success(f"**Result: {prediction_str}**")
        
        st.metric("Confidence (P(PD))", f"{probability:.1%}")
        
        if true_label is not None:
            if prediction == true_label:
                st.write("✅ **Correct Prediction**")
            else:
                st.write("❌ **Incorrect Prediction**")
        
        st.markdown("---")
        st.subheader("Extracted Features")
        
        # Display feature values
        for name, value in features_dict.items():
            st.write(f"**{name}:** `{value:.4f}`")

else:
    st.info("👈 Please upload an image or pick a random dataset image from the sidebar.")

st.markdown("---")
st.caption("⚠️ This is a screening aid, not a diagnostic device. Handcrafted baseline demo.")
