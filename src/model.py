"""
CNN models for spiral image classification.

Two builders, both used in production:
    - build_model():         3-block CNN (v4 - no BatchNorm, no L2). Trains
                             from scratch on oversampled balanced data.
    - build_mobilenetv2():   MobileNetV2 transfer learning (frozen ImageNet
                             base + small custom head). The architecture
                             mirrors app/streamlit_app.py::load_mobilenet_model()
                             exactly so saved .weights.h5 files load cleanly
                             in the Streamlit demo without a custom_objects scope.

Design notes:
    Initial versions (v1-v3) of build_model() used BatchNorm + L2 + class_weight
    + augmentation. All collapsed to "predict all PD" (100% recall, 0% specificity)
    because:
        1. Class imbalance (420 PD vs 212 healthy) - model took shortcut
        2. BatchNorm noise on small batches pushed model toward collapse
        3. Class weighting amplified the collapse

    v4 fix (in src/train.py): oversample healthy images to 50/50 balance, then
    train this simple architecture (no BN, no L2). Achieves 81.5% accuracy and
    97.9% recall on patient-level 5-fold CV.

    MobileNetV2 (transfer learning, frozen base) further improves to 89.4%
    accuracy / 93.6% recall - pretrained ImageNet features transfer well to
    spiral images.

Architectures:
    v4 CNN (~110K params, all trainable):
        Input (256, 256, 1)
        -> Conv(32, 3x3) - ReLU - MaxPool(2x2) - Dropout(0.25)
        -> Conv(64, 3x3) - ReLU - MaxPool(2x2) - Dropout(0.25)
        -> Conv(128, 3x3) - ReLU - MaxPool(2x2)
        -> GlobalAveragePooling2D
        -> Dense(64) - ReLU - Dropout(0.4)
        -> Dense(1, Sigmoid)

    MobileNetV2 (~3.5M total, only ~8K trainable):
        Input (256, 256, 3)   <- grayscale replicated to 3 channels
        -> MobileNetV2 base (pretrained ImageNet, FROZEN)
        -> GlobalAveragePooling2D
        -> Dense(64) - ReLU - Dropout(0.4)
        -> Dense(1, Sigmoid)
"""

import tensorflow as tf
from tensorflow.keras import layers, models


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INPUT_SHAPE = (256, 256, 1)
MOBILENET_INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = 1
DROPOUT_CONV = 0.25
DROPOUT_DENSE = 0.40
MOBILENET_DROPOUT = 0.40
DEFAULT_LR = 1e-4


# ---------------------------------------------------------------------------
# CNN v4 - working version (no BatchNorm, no L2)
# ---------------------------------------------------------------------------

def build_model(
    input_shape=INPUT_SHAPE,
    dropout_conv=DROPOUT_CONV,
    dropout_dense=DROPOUT_DENSE,
    learning_rate=DEFAULT_LR,
):
    """Build the 3-block CNN model (v4 - no BatchNorm, no L2).

    This version works with oversampled balanced training data. Previous
    versions with BatchNorm + L2 collapsed to "predict all PD".

    Args:
        input_shape: image shape, default (256, 256, 1) grayscale
        dropout_conv: dropout after conv blocks (default 0.25)
        dropout_dense: dropout in dense layer (default 0.40)
        learning_rate: Adam learning rate (default 1e-4)

    Returns:
        tensorflow.keras.Model, compiled and ready to train
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input"),

        # Block 1: 32 filters
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1"),
        layers.MaxPooling2D((2, 2), name="pool1"),
        layers.Dropout(dropout_conv, name="drop1"),

        # Block 2: 64 filters
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2"),
        layers.MaxPooling2D((2, 2), name="pool2"),
        layers.Dropout(dropout_conv, name="drop2"),

        # Block 3: 128 filters
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3"),
        layers.MaxPooling2D((2, 2), name="pool3"),

        # Head
        layers.GlobalAveragePooling2D(name="gap"),
        layers.Dense(64, activation="relu", name="dense1"),
        layers.Dropout(dropout_dense, name="drop_dense"),
        layers.Dense(NUM_CLASSES, activation="sigmoid", name="output"),
    ], name="spiral_cnn_v4")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# MobileNetV2 - transfer learning (frozen ImageNet base + custom head)
# ---------------------------------------------------------------------------

def build_mobilenetv2(
    input_shape=MOBILENET_INPUT_SHAPE,
    dropout_rate=MOBILENET_DROPOUT,
    learning_rate=DEFAULT_LR,
):
    """Build the MobileNetV2 transfer learning model.

    Architecture mirrors app/streamlit_app.py::load_mobilenet_model() so the
    saved .weights.h5 file loads cleanly in the Streamlit demo. The ImageNet-
    pretrained base is frozen - only the small custom head trains (~8K params).

    Caller is responsible for converting grayscale input (256, 256, 1) to
    3-channel (256, 256, 3) via np.repeat(x, 3, axis=-1) before fitting /
    predicting.

    Args:
        input_shape: image shape, must be (H, W, 3) for MobileNetV2
        dropout_rate: dropout in dense layer (default 0.40)
        learning_rate: Adam learning rate (default 1e-4)

    Returns:
        tensorflow.keras.Model, compiled and ready to train
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # freeze pretrained layers

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(1, activation="sigmoid"),
    ], name="spiral_mobilenetv2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ---------------------------------------------------------------------------
# Quick test: build and show summary
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import numpy as np

    print("=" * 70)
    print("CNN v4 (spiral_cnn_v4)")
    print("=" * 70)
    m1 = build_model()
    m1.summary()
    print(f"\nTotal parameters: {m1.count_params():,}")

    print("\n" + "=" * 70)
    print("MobileNetV2 (spiral_mobilenetv2)")
    print("=" * 70)
    m2 = build_mobilenetv2()
    m2.summary()
    print(f"\nTotal parameters:     {m2.count_params():,}")
    trainable = int(np.sum([np.prod(w.shape) for w in m2.trainable_weights]))
    print(f"Trainable parameters: {trainable:,}  (frozen base + trainable head)")
    print("\nArchitecture: MobileNetV2(frozen)-GAP-Dense(64)-Dropout(0.4)-Sigmoid")
    print("Input shape: (256, 256, 3) - grayscale replicated to 3 channels")
