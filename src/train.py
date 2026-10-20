"""
Training script for the handcrafted-features baseline, the CNN v4, AND
the MobileNetV2 transfer-learning model.

All three models are trained with patient-level 5-fold StratifiedGroupKFold
cross-validation. The two neural nets additionally use oversampling of the
healthy class to 50/50 balance in each training fold (this fixed the
"predict all PD" collapse in CNN v1-v3). Reports per-fold and pooled
out-of-fold (OOF) accuracy.

Usage:
    python src/train.py --model baseline    # LR only
    python src/train.py --model cnn         # CNN v4 only
    python src/train.py --model mobilenet   # MobileNetV2 only
    python src/train.py --model both        # LR + CNN (legacy alias)
    python src/train.py --model all         # all three (default)

Outputs (in data/processed/):
    - oof_predictions_handcrafted.npy
    - oof_predictions_cnn.npy
    - oof_probabilities_cnn.npy
    - oof_predictions_mobilenet.npy
    - oof_probabilities_mobilenet.npy
    (used by src/evaluate.py for McNemar tests + 3-way ensemble)
"""

import os
import sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.utils import resample

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import (
    load_newhandpd,
    patient_level_split,
    verify_no_patient_overlap,
)
from src.features import extract_features_for_dataset


# ===========================================================================
# PART 1: HANDCRAFTED BASELINE (Logistic Regression)
# ===========================================================================

def train_handcrafted_baseline(features, labels, patient_ids, n_splits=5):
    """Train LR baseline with patient-level 5-fold CV.

    Args:
        features: numpy array (632, 5)
        labels: numpy array (632,)
        patient_ids: numpy array (632,)
        n_splits: number of CV folds

    Returns:
        dict with OOF predictions, fold metrics, pooled metrics
    """
    print(f"\nRunning LR baseline {n_splits}-fold patient-level CV...")

    splits = patient_level_split(features, labels, patient_ids,
                                 n_splits=n_splits, random_state=42)

    oof_predictions = np.zeros(len(labels), dtype=int)
    oof_probabilities = np.zeros(len(labels), dtype=float)
    oof_test_mask = np.zeros(len(labels), dtype=bool)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        verify_no_patient_overlap(patient_ids, train_idx, test_idx)

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)
        probabilities = model.predict_proba(X_test_scaled)[:, 1]  # P(PD)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        cm = confusion_matrix(y_test, predictions)

        print(f"  Fold {fold_idx + 1}: acc={accuracy:.4f}, prec={precision:.4f}, "
              f"rec={recall:.4f}, f1={f1:.4f}")

        oof_predictions[test_idx] = predictions
        oof_probabilities[test_idx] = probabilities
        oof_test_mask[test_idx] = True
        fold_metrics.append({
            "fold": fold_idx + 1, "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "confusion_matrix": cm,
        })

    assert oof_test_mask.all(), "Some images were never predicted"

    pooled_accuracy = accuracy_score(labels, oof_predictions)
    pooled_precision = precision_score(labels, oof_predictions, zero_division=0)
    pooled_recall = recall_score(labels, oof_predictions, zero_division=0)
    pooled_f1 = f1_score(labels, oof_predictions, zero_division=0)
    pooled_cm = confusion_matrix(labels, oof_predictions)

    fold_accuracies = [m["accuracy"] for m in fold_metrics]
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    return {
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
        "fold_metrics": fold_metrics,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "pooled_accuracy": pooled_accuracy,
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "pooled_f1": pooled_f1,
        "pooled_confusion_matrix": pooled_cm,
    }


def print_results(results, feature_names):
    """Print baseline results."""
    print("\n" + "=" * 70)
    print("HANDCRAFTED BASELINE RESULTS - Patient-Level 5-Fold CV")
    print("=" * 70)

    print(f"\n--- Per-Fold Results ---")
    print(f"{'Fold':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    for m in results["fold_metrics"]:
        print(f"{m['fold']:<6} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
              f"{m['recall']:<12.4f} {m['f1']:<12.4f}")

    print(f"\n--- Summary ---")
    print(f"Mean accuracy:  {results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f}")
    print(f"Pooled OOF acc: {results['pooled_accuracy']:.4f}")
    print(f"Pooled recall:  {results['pooled_recall']:.4f}")
    print(f"Pooled F1:      {results['pooled_f1']:.4f}")


def save_predictions(results):
    """Save OOF predictions + probabilities for McNemar test and ensemble.

    Saves under both the legacy 'handcrafted' name and the 'lr' name that
    evaluate.py expects, so either version of evaluate.py finds the files.
    Probabilities (P(PD)) are needed for the 3-way ensemble averaging.
    """
    os.makedirs("data/processed", exist_ok=True)
    # Legacy names (backward compat with old docs)
    np.save("data/processed/oof_predictions_handcrafted.npy", results["oof_predictions"])
    # New names that evaluate.py looks for
    np.save("data/processed/oof_predictions_lr.npy", results["oof_predictions"])
    np.save("data/processed/oof_probabilities_lr.npy", results["oof_probabilities"])
    print(f"  Saved to data/processed/oof_predictions_lr.npy")
    print(f"  Saved to data/processed/oof_probabilities_lr.npy")
    print(f"  Saved to data/processed/oof_predictions_handcrafted.npy (legacy)")


def analyze_feature_importance(features, labels, patient_ids, feature_names):
    """Analyze which features matter most using LR coefficients."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(features_scaled, labels)

    print(f"\n--- Feature Importance (LR coefficients) ---")
    coefs = model.coef_[0]
    ranked = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
    for i, (name, coef) in enumerate(ranked):
        print(f"  {i + 1}. {name:<30} {coef:+.4f}")


# ===========================================================================
# PART 2: CNN v4 TRAINING (oversampling, no BatchNorm)
# ===========================================================================

def train_cnn_one_fold(X_train, y_train, X_test, y_test, fold_idx,
                       epochs=60, batch_size=16, learning_rate=1e-4):
    """Train the CNN on one fold with oversampled balanced data.

    v4 design (working - no collapse):
        - Oversample healthy class to match PD count (50/50 training)
        - NO BatchNorm (destabilized training on small batches)
        - NO augmentation (adds noise when model is already struggling)
        - NO L2 regularization (over-constrained the model)
        - NO class_weight (backfired - amplified collapse)
        - Fresh model per fold

    Previous versions (v1-v3) all collapsed to "predict all PD" because:
        1. Class imbalance (420 PD vs 212 healthy) - shortcut to 66% accuracy
        2. BatchNorm noise on small batches pushed toward collapse
        3. Class weighting amplified the collapse

    Args:
        X_train: training images, shape (n_train, 256, 256, 1)
        y_train: training labels, shape (n_train,)
        X_test: test images, shape (n_test, 256, 256, 1)
        y_test: test labels, shape (n_test,)
        fold_idx: which fold (for printing)
        epochs: number of epochs (default 60)
        batch_size: training batch size (default 16)
        learning_rate: Adam learning rate (default 1e-4)

    Returns:
        dict with predictions, probabilities, metrics, history
    """
    import tensorflow as tf
    from src.model import build_model

    # 1. OVERSAMPLE healthy class to match PD count
    pd_idx = np.where(y_train == 1)[0]
    healthy_idx = np.where(y_train == 0)[0]
    n_pd = len(pd_idx)

    healthy_upsampled = resample(healthy_idx, n_samples=n_pd,
                                 random_state=42, replace=True)
    balanced_idx = np.concatenate([pd_idx, healthy_upsampled])

    X_train_bal = X_train[balanced_idx]
    y_train_bal = y_train[balanced_idx]

    # Shuffle
    shuffle_idx = np.random.permutation(len(balanced_idx))
    X_train_bal = X_train_bal[shuffle_idx]
    y_train_bal = y_train_bal[shuffle_idx]

    print(f"  Fold {fold_idx} balanced: PD={n_pd}, Healthy={n_pd} (oversampled)")

    # 2. Build fresh model (v4 - no BatchNorm, no L2)
    model = build_model(learning_rate=learning_rate)

    # 3. Train (NO class_weight - oversampling handles balance)
    print(f"  Training fold {fold_idx} (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(
        X_train_bal, y_train_bal,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    # 4. Predict on ORIGINAL test set (never oversampled)
    probabilities = model.predict(X_test, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)

    # 5. Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    # Collapse check
    if recall == 1.0 and cm[0, 0] == 0:
        print(f"  WARNING: Model collapsed (predicting all PD)")
    else:
        print(f"  LEARNING: TN={cm[0,0]} healthy correctly identified")

    print(f"  Fold {fold_idx}: acc={accuracy:.4f}, prec={precision:.4f}, "
          f"rec={recall:.4f}, f1={f1:.4f}")
    print(f"  Train acc={final_train_acc:.3f}, Val acc={final_val_acc:.3f}")
    print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    del model
    tf.keras.backend.clear_session()

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "history": history.history,
    }


def train_cnn_with_cv(images, labels, patient_ids, n_splits=5, epochs=60, batch_size=16):
    """Train CNN with patient-level 5-fold CV + oversampling.

    Same protocol as baseline, but with oversampled balanced training data
    and the v4 CNN model (no BatchNorm).

    Args:
        images: numpy array (632, 256, 256, 1)
        labels: numpy array (632,)
        patient_ids: numpy array (632,)
        n_splits: number of CV folds
        epochs: epochs per fold
        batch_size: training batch size

    Returns:
        dict with OOF predictions, fold metrics, pooled metrics
    """
    print(f"\nRunning CNN v4 {n_splits}-fold patient-level CV with oversampling...")

    splits = patient_level_split(images, labels, patient_ids,
                                 n_splits=n_splits, random_state=42)

    oof_predictions = np.zeros(len(labels), dtype=int)
    oof_probabilities = np.zeros(len(labels), dtype=float)
    oof_test_mask = np.zeros(len(labels), dtype=bool)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        verify_no_patient_overlap(patient_ids, train_idx, test_idx)

        X_train, X_test = images[train_idx], images[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        result = train_cnn_one_fold(
            X_train, y_train, X_test, y_test, fold_idx + 1,
            epochs=epochs, batch_size=batch_size,
        )

        oof_predictions[test_idx] = result["predictions"]
        oof_probabilities[test_idx] = result["probabilities"]
        oof_test_mask[test_idx] = True

        fold_metrics.append({
            "fold": fold_idx + 1,
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "confusion_matrix": result["confusion_matrix"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

    assert oof_test_mask.all(), "Some images were never predicted"

    pooled_accuracy = accuracy_score(labels, oof_predictions)
    pooled_precision = precision_score(labels, oof_predictions, zero_division=0)
    pooled_recall = recall_score(labels, oof_predictions, zero_division=0)
    pooled_f1 = f1_score(labels, oof_predictions, zero_division=0)
    pooled_cm = confusion_matrix(labels, oof_predictions)

    fold_accuracies = [m["accuracy"] for m in fold_metrics]
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    return {
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
        "fold_metrics": fold_metrics,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "pooled_accuracy": pooled_accuracy,
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "pooled_f1": pooled_f1,
        "pooled_confusion_matrix": pooled_cm,
    }


def print_cnn_results(results):
    """Print CNN results."""
    print("\n" + "=" * 70)
    print("CNN RESULTS - Patient-Level 5-Fold CV (v4 with oversampling)")
    print("=" * 70)

    print(f"\n--- Per-Fold Results ---")
    print(f"{'Fold':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    for m in results["fold_metrics"]:
        print(f"{m['fold']:<6} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
              f"{m['recall']:<12.4f} {m['f1']:<12.4f}")

    print(f"\n--- Summary ---")
    print(f"Mean accuracy:  {results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f}")
    print(f"Pooled OOF acc: {results['pooled_accuracy']:.4f}")
    print(f"Pooled recall:  {results['pooled_recall']:.4f}")
    print(f"Pooled F1:      {results['pooled_f1']:.4f}")

    cm = results["pooled_confusion_matrix"]
    print(f"\n--- Confusion Matrix ---")
    print(f"                  Pred Healthy  Predicted PD")
    print(f"  Actual Healthy    {cm[0,0]:>5} (TN)        {cm[0,1]:>5} (FP)")
    print(f"  Actual PD         {cm[1,0]:>5} (FN)        {cm[1,1]:>5} (TP)")
    print(f"\n  Missed PD (FN): {cm[1,0]}")
    print(f"  False alarms (FP): {cm[0,1]}")


def save_cnn_predictions(results):
    """Save CNN OOF predictions for McNemar test."""
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/oof_predictions_cnn.npy", results["oof_predictions"])
    np.save("data/processed/oof_probabilities_cnn.npy", results["oof_probabilities"])
    print(f"  Saved to data/processed/oof_predictions_cnn.npy")
    print(f"  Saved to data/processed/oof_probabilities_cnn.npy")


# ===========================================================================
# PART 3: MOBILENETV2 TRAINING (transfer learning, frozen base)
# ===========================================================================

def train_mobilenet_one_fold(X_train, y_train, X_test, y_test, fold_idx,
                             epochs=60, batch_size=16, learning_rate=1e-4):
    """Train MobileNetV2 on one fold with oversampled balanced data.

    Transfer learning with frozen ImageNet-pretrained base. Only the small
    custom head (~8K params) is trained. Same oversampling protocol as the
    CNN v4 - oversample healthy class to 50/50, no class_weight, no
    augmentation.

    Grayscale (256, 256, 1) inputs are replicated to (256, 256, 3) via
    np.repeat(x, 3, axis=-1) so MobileNetV2's pretrained RGB weights can
    fire on them. The same conversion is done at inference time in
    app/streamlit_app.py.

    Args:
        X_train: training images (n_train, 256, 256, 1) grayscale
        y_train: training labels (n_train,)
        X_test: test images (n_test, 256, 256, 1) grayscale
        y_test: test labels (n_test,)
        fold_idx: fold number for printing
        epochs: training epochs (default 60)
        batch_size: training batch size (default 16)
        learning_rate: Adam learning rate (default 1e-4)

    Returns:
        dict with predictions, probabilities, metrics, history
    """
    import tensorflow as tf
    from src.model import build_mobilenetv2

    # 1. OVERSAMPLE healthy class to match PD count
    pd_idx = np.where(y_train == 1)[0]
    healthy_idx = np.where(y_train == 0)[0]
    n_pd = len(pd_idx)

    healthy_upsampled = resample(healthy_idx, n_samples=n_pd,
                                 random_state=42, replace=True)
    balanced_idx = np.concatenate([pd_idx, healthy_upsampled])

    X_train_bal = X_train[balanced_idx]
    y_train_bal = y_train[balanced_idx]

    # Shuffle
    shuffle_idx = np.random.permutation(len(balanced_idx))
    X_train_bal = X_train_bal[shuffle_idx]
    y_train_bal = y_train_bal[shuffle_idx]

    # 2. CONVERT grayscale -> RGB (3 channels) for MobileNetV2
    #    Same trick as app/streamlit_app.py: np.repeat(x, 3, axis=-1)
    X_train_rgb = np.repeat(X_train_bal, 3, axis=-1)
    X_test_rgb = np.repeat(X_test, 3, axis=-1)

    print(f"  Fold {fold_idx} balanced: PD={n_pd}, Healthy={n_pd} (oversampled)")
    print(f"  Fold {fold_idx} RGB: train={X_train_rgb.shape}, test={X_test_rgb.shape}")

    # 3. Build fresh model (frozen ImageNet base + trainable head)
    model = build_mobilenetv2(learning_rate=learning_rate)

    # 4. Train (only head trains - base is frozen)
    print(f"  Training fold {fold_idx} (epochs={epochs}, batch_size={batch_size})...")
    history = model.fit(
        X_train_rgb, y_train_bal,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    # 5. Predict on ORIGINAL test set (never oversampled)
    probabilities = model.predict(X_test_rgb, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)

    # 6. Metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]

    # Collapse check
    if recall == 1.0 and cm[0, 0] == 0:
        print(f"  WARNING: Model collapsed (predicting all PD)")
    else:
        print(f"  LEARNING: TN={cm[0,0]} healthy correctly identified")

    print(f"  Fold {fold_idx}: acc={accuracy:.4f}, prec={precision:.4f}, "
          f"rec={recall:.4f}, f1={f1:.4f}")
    print(f"  Train acc={final_train_acc:.3f}, Val acc={final_val_acc:.3f}")
    print(f"  Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    del model
    tf.keras.backend.clear_session()

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "history": history.history,
    }


def train_mobilenet_with_cv(images, labels, patient_ids, n_splits=5,
                            epochs=60, batch_size=16):
    """Train MobileNetV2 with patient-level 5-fold CV + oversampling.

    Same protocol as train_cnn_with_cv, but with MobileNetV2 (frozen
    ImageNet base + trainable head). Grayscale inputs are replicated to
    RGB inside train_mobilenet_one_fold.

    Args:
        images: numpy array (632, 256, 256, 1) grayscale
        labels: numpy array (632,)
        patient_ids: numpy array (632,)
        n_splits: number of CV folds
        epochs: epochs per fold
        batch_size: training batch size

    Returns:
        dict with OOF predictions, fold metrics, pooled metrics
    """
    print(f"\nRunning MobileNetV2 {n_splits}-fold patient-level CV with oversampling...")

    splits = patient_level_split(images, labels, patient_ids,
                                 n_splits=n_splits, random_state=42)

    oof_predictions = np.zeros(len(labels), dtype=int)
    oof_probabilities = np.zeros(len(labels), dtype=float)
    oof_test_mask = np.zeros(len(labels), dtype=bool)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        verify_no_patient_overlap(patient_ids, train_idx, test_idx)

        X_train, X_test = images[train_idx], images[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        result = train_mobilenet_one_fold(
            X_train, y_train, X_test, y_test, fold_idx + 1,
            epochs=epochs, batch_size=batch_size,
        )

        oof_predictions[test_idx] = result["predictions"]
        oof_probabilities[test_idx] = result["probabilities"]
        oof_test_mask[test_idx] = True

        fold_metrics.append({
            "fold": fold_idx + 1,
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "confusion_matrix": result["confusion_matrix"],
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

    assert oof_test_mask.all(), "Some images were never predicted"

    pooled_accuracy = accuracy_score(labels, oof_predictions)
    pooled_precision = precision_score(labels, oof_predictions, zero_division=0)
    pooled_recall = recall_score(labels, oof_predictions, zero_division=0)
    pooled_f1 = f1_score(labels, oof_predictions, zero_division=0)
    pooled_cm = confusion_matrix(labels, oof_predictions)

    fold_accuracies = [m["accuracy"] for m in fold_metrics]
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    return {
        "oof_predictions": oof_predictions,
        "oof_probabilities": oof_probabilities,
        "fold_metrics": fold_metrics,
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "pooled_accuracy": pooled_accuracy,
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "pooled_f1": pooled_f1,
        "pooled_confusion_matrix": pooled_cm,
    }


def print_mobilenet_results(results):
    """Print MobileNetV2 results."""
    print("\n" + "=" * 70)
    print("MOBILENETV2 RESULTS - Patient-Level 5-Fold CV (oversampling)")
    print("=" * 70)

    print(f"\n--- Per-Fold Results ---")
    print(f"{'Fold':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)
    for m in results["fold_metrics"]:
        print(f"{m['fold']:<6} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
              f"{m['recall']:<12.4f} {m['f1']:<12.4f}")

    print(f"\n--- Summary ---")
    print(f"Mean accuracy:  {results['mean_accuracy']:.4f} +/- {results['std_accuracy']:.4f}")
    print(f"Pooled OOF acc: {results['pooled_accuracy']:.4f}")
    print(f"Pooled recall:  {results['pooled_recall']:.4f}")
    print(f"Pooled F1:      {results['pooled_f1']:.4f}")

    cm = results["pooled_confusion_matrix"]
    print(f"\n--- Confusion Matrix ---")
    print(f"                  Pred Healthy  Predicted PD")
    print(f"  Actual Healthy    {cm[0,0]:>5} (TN)        {cm[0,1]:>5} (FP)")
    print(f"  Actual PD         {cm[1,0]:>5} (FN)        {cm[1,1]:>5} (TP)")
    print(f"\n  Missed PD (FN): {cm[1,0]}")
    print(f"  False alarms (FP): {cm[0,1]}")


def save_mobilenet_predictions(results):
    """Save MobileNetV2 OOF predictions for McNemar test + ensemble."""
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/oof_predictions_mobilenet.npy", results["oof_predictions"])
    np.save("data/processed/oof_probabilities_mobilenet.npy", results["oof_probabilities"])
    print(f"  Saved to data/processed/oof_predictions_mobilenet.npy")
    print(f"  Saved to data/processed/oof_probabilities_mobilenet.npy")


# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train models for Parkinson's spiral screening")
    parser.add_argument("--model", choices=["baseline", "cnn", "mobilenet", "both", "all"],
                        default="all",
                        help="Which model to train (default: all)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="CNN/MobileNet epochs (default 60)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default 16)")
    args = parser.parse_args()

    print("=" * 70)
    print("Loading dataset...")
    print("=" * 70)
    images, labels, patient_ids, filenames = load_newhandpd("data/raw/Merged")

    run_baseline = args.model in ["baseline", "both", "all"]
    run_cnn = args.model in ["cnn", "both", "all"]
    run_mobilenet = args.model in ["mobilenet", "all"]

    baseline_results = None
    cnn_results = None
    mobilenet_results = None

    # --- Baseline ---
    if run_baseline:
        print("\n" + "=" * 70)
        print("HANDCRAFTED BASELINE - Patient-Level 5-Fold CV")
        print("=" * 70)

        # Extract the 5 handcrafted features from images (slow first run,
        # cached as .npy on subsequent runs).
        import os as _os
        feat_cache = "data/processed/features_handcrafted.npy"
        names_cache = "data/processed/feature_names.npy"
        if _os.path.exists(feat_cache) and _os.path.exists(names_cache):
            print("  Loading cached features...")
            features = np.load(feat_cache)
            feature_names = np.load(names_cache, allow_pickle=True).tolist()
        else:
            print("  Extracting handcrafted features (first run, slow)...")
            features, feature_names = extract_features_for_dataset(images)
            _os.makedirs("data/processed", exist_ok=True)
            np.save(feat_cache, features)
            np.save(names_cache, np.array(feature_names, dtype=object))
            print(f"  Cached to {feat_cache}")
        assert len(features) == len(labels)

        baseline_results = train_handcrafted_baseline(features, labels, patient_ids, n_splits=5)
        print_results(baseline_results, feature_names)
        save_predictions(baseline_results)
        analyze_feature_importance(features, labels, patient_ids, feature_names)

        print(f"\nBaseline accuracy: {baseline_results['pooled_accuracy']:.1%}")

    # --- CNN v4 ---
    if run_cnn:
        print("\n" + "=" * 70)
        print("CNN TRAINING (v4) - Patient-Level 5-Fold CV with Oversampling")
        print("=" * 70)

        print(f"\nEpochs: {args.epochs}, Batch size: {args.batch_size}")
        print("Oversampling: healthy class duplicated to match PD count")
        print("Architecture: Conv(32)-Conv(64)-Conv(128)-GAP-Dense(64)-Sigmoid")
        print("No BatchNorm, no L2, no augmentation (v4 working version)")

        cnn_results = train_cnn_with_cv(
            images, labels, patient_ids,
            n_splits=5, epochs=args.epochs, batch_size=args.batch_size,
        )

        print_cnn_results(cnn_results)
        save_cnn_predictions(cnn_results)

        print(f"\nCNN accuracy: {cnn_results['pooled_accuracy']:.1%}")

    # --- MobileNetV2 ---
    if run_mobilenet:
        print("\n" + "=" * 70)
        print("MOBILENETV2 TRAINING - Patient-Level 5-Fold CV with Oversampling")
        print("=" * 70)

        print(f"\nEpochs: {args.epochs}, Batch size: {args.batch_size}")
        print("Oversampling: healthy class duplicated to match PD count")
        print("Architecture: MobileNetV2(frozen)-GAP-Dense(64)-Dropout(0.4)-Sigmoid")
        print("Pretrained ImageNet base, only custom head trains (~8K params)")
        print("Grayscale input (256,256,1) replicated to (256,256,3) for MobileNetV2")

        mobilenet_results = train_mobilenet_with_cv(
            images, labels, patient_ids,
            n_splits=5, epochs=args.epochs, batch_size=args.batch_size,
        )

        print_mobilenet_results(mobilenet_results)
        save_mobilenet_predictions(mobilenet_results)

        print(f"\nMobileNetV2 accuracy: {mobilenet_results['pooled_accuracy']:.1%}")

    # --- Summary ---
    if args.model in ["both", "all"]:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        if baseline_results:
            print(f"  Handcrafted baseline: {baseline_results['pooled_accuracy']:.1%} "
                  f"(recall {baseline_results['pooled_recall']:.1%})")
        if cnn_results:
            print(f"  CNN v4:               {cnn_results['pooled_accuracy']:.1%} "
                  f"(recall {cnn_results['pooled_recall']:.1%})")
        if mobilenet_results:
            print(f"  MobileNetV2:          {mobilenet_results['pooled_accuracy']:.1%} "
                  f"(recall {mobilenet_results['pooled_recall']:.1%})")

        if cnn_results and baseline_results:
            diff = cnn_results["pooled_accuracy"] - baseline_results["pooled_accuracy"]
            print(f"  CNN - Baseline:       {diff:+.1%}")
        if mobilenet_results and cnn_results:
            diff = mobilenet_results["pooled_accuracy"] - cnn_results["pooled_accuracy"]
            print(f"  MobileNet - CNN:      {diff:+.1%}")

        print(f"\nNext: Run McNemar test + 3-way ensemble (src/evaluate.py) "
              f"for significance + ROC curves.")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
