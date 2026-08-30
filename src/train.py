"""
Training script for the handcrafted-features baseline.

Trains a logistic regression on the 5 handcrafted features using patient-level
5-fold StratifiedGroupKFold cross-validation. Reports per-fold and pooled
out-of-fold (OOF) accuracy with mean ± std.

This is the FIRST honest accuracy number of the project. Expected: 70-80%.

Why logistic regression?
    - Simple, interpretable, fast
    - Standard baseline for binary classification
    - Won't overfit on small feature sets (5 features, 264 samples)
    - The literature predicts handcrafted features may match the CNN on
      sparse spiral data — if LR gets ~75% and CNN gets ~82%, the McNemar
      test will tell us if that difference is significant

Methodology:
    Patient-level CV means each patient's drawings go entirely into either
    training or testing, never both. This prevents the data leakage that
    inflates accuracy by 30-55 percentage points per PMC 2021 (PMC8604922).

Usage:
    python src/train.py

Outputs:
    - Per-fold accuracy, precision, recall, F1
    - Pooled OOF accuracy ± std
    - Saved OOF predictions at data/processed/oof_predictions_handcrafted.npy
      (used later for McNemar test against the CNN)
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

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import (
    load_newhandpd,
    patient_level_split,
    verify_no_patient_overlap,
)


# ---------------------------------------------------------------------------
# Step 1: Load features (saved by src/features.py)
# ---------------------------------------------------------------------------

def load_features():
    """Load the handcrafted features saved by src/features.py.

    Returns:
        feature_matrix: np.array of shape (264, 5) — features for each image
        feature_names: list of 5 feature name strings
    """
    features_path = "data/processed/features_handcrafted.npy"
    names_path = "data/processed/feature_names.npy"

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Features not found at {features_path}. "
            "Run `python src/features.py` first to generate them."
        )

    feature_matrix = np.load(features_path)
    feature_names = np.load(names_path, allow_pickle=True).tolist()

    print(f"Loaded features: {feature_matrix.shape}")
    print(f"Feature names: {feature_names}")
    return feature_matrix, feature_names


# ---------------------------------------------------------------------------
# Step 2: Train one fold of logistic regression
# ---------------------------------------------------------------------------

def train_one_fold(X_train, y_train, X_test, y_test, fold_idx):
    """Train logistic regression on one fold and return predictions + metrics.

    Includes feature standardization (z-score normalization) — critical for
    logistic regression because features have very different scales:
        - intersection_count: 0-300
        - stroke_entropy: 0.97-0.99
        - mean_squared_displacement: 0.12-0.15

    Without standardization, the feature with the largest scale dominates.

    Args:
        X_train: training features, shape (n_train, 5)
        y_train: training labels, shape (n_train,)
        X_test: test features, shape (n_test, 5)
        y_test: test labels, shape (n_test,)
        fold_idx: which fold (for printing)

    Returns:
        dict with:
            - predictions: binary predictions for test set
            - probabilities: predicted probabilities for test set
            - accuracy, precision, recall, f1: scalars
            - confusion_matrix: 2x2 array
    """
    # Standardize features: zero mean, unit variance
    # Fit on TRAINING data only, then transform both train and test
    # (Critical: fitting on test data would be leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train logistic regression
    # - class_weight='balanced': automatically weight minority class higher
    #   (our dataset is ~53/47, so this has minor effect, but it's good practice)
    # - max_iter=1000: increase from default 100 to ensure convergence
    # - random_state=42: reproducibility
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    predictions = model.predict(X_test_scaled)
    probabilities = model.predict_proba(X_test_scaled)[:, 1]  # P(class=1)

    # Compute metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    cm = confusion_matrix(y_test, predictions)

    print(f"  Fold {fold_idx}: "
          f"acc={accuracy:.4f}, prec={precision:.4f}, "
          f"rec={recall:.4f}, f1={f1:.4f}")
    print(f"           Confusion: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    return {
        "predictions": predictions,
        "probabilities": probabilities,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "model": model,
        "scaler": scaler,
    }


# ---------------------------------------------------------------------------
# Step 3: Full 5-fold patient-level CV
# ---------------------------------------------------------------------------

def train_handcrafted_baseline(features, labels, patient_ids, n_splits=5, random_state=42):
    """Train logistic regression with patient-level 5-fold CV.

    This is the MAIN function. It:
        1. Creates the 5-fold patient-level split
        2. For each fold: trains LR, predicts on test, collects OOF predictions
        3. Pools all OOF predictions to compute overall accuracy
        4. Reports mean ± std across folds

    Args:
        features: np.array of shape (264, 5)
        labels: np.array of shape (264,)
        patient_ids: np.array of shape (264,)
        n_splits: number of CV folds (default 5)
        random_state: for reproducibility

    Returns:
        dict with:
            - oof_predictions: pooled predictions for ALL images (264,)
            - oof_probabilities: pooled probabilities for ALL images (264,)
            - fold_metrics: list of per-fold metric dicts
            - mean_accuracy, std_accuracy: scalars
            - pooled_accuracy: scalar (computed on pooled OOF)
    """
    print(f"\nRunning {n_splits}-fold patient-level cross-validation...")

    # Get the 5-fold splits
    splits = patient_level_split(features, labels, patient_ids,
                                 n_splits=n_splits, random_state=random_state)

    # Storage for OOF predictions
    # Each image will be predicted exactly once (when it's in the test fold)
    oof_predictions = np.zeros(len(labels), dtype=int)
    oof_probabilities = np.zeros(len(labels), dtype=float)
    oof_test_mask = np.zeros(len(labels), dtype=bool)  # track which images we've predicted

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Verify no patient overlap (defensive — should never fail)
        verify_no_patient_overlap(patient_ids, train_idx, test_idx)

        # Split data
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        # Train and predict
        result = train_one_fold(X_train, y_train, X_test, y_test, fold_idx + 1)

        # Store OOF predictions
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

    # Verify all images were predicted exactly once
    assert oof_test_mask.all(), "Some images were never predicted — check the splits"
    assert oof_test_mask.sum() == len(labels), "Some images predicted more than once"

    # Compute pooled accuracy (on all OOF predictions together)
    pooled_accuracy = accuracy_score(labels, oof_predictions)
    pooled_precision = precision_score(labels, oof_predictions, zero_division=0)
    pooled_recall = recall_score(labels, oof_predictions, zero_division=0)
    pooled_f1 = f1_score(labels, oof_predictions, zero_division=0)
    pooled_cm = confusion_matrix(labels, oof_predictions)

    # Compute mean ± std across folds
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


# ---------------------------------------------------------------------------
# Step 4: Print results nicely
# ---------------------------------------------------------------------------

def print_results(results, feature_names):
    """Print the results in a clear, report-ready format.

    Args:
        results: dict returned by train_handcrafted_baseline
        feature_names: list of feature name strings
    """
    print("\n" + "=" * 70)
    print("HANDCRAFTED BASELINE RESULTS")
    print("=" * 70)

    # Per-fold results
    print(f"\n--- Per-Fold Results ---")
    print(f"{'Fold':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Train':<8} {'Test':<8}")
    print("-" * 70)
    for m in results["fold_metrics"]:
        print(f"{m['fold']:<6} {m['accuracy']:<12.4f} {m['precision']:<12.4f} "
              f"{m['recall']:<12.4f} {m['f1']:<12.4f} {m['n_train']:<8} {m['n_test']:<8}")

    # Summary stats
    print(f"\n--- Summary ---")
    print(f"Mean accuracy across folds: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
    print(f"Pooled OOF accuracy:        {results['pooled_accuracy']:.4f}")
    print(f"Pooled precision:           {results['pooled_precision']:.4f}")
    print(f"Pooled recall:              {results['pooled_recall']:.4f}  (critical for screening)")
    print(f"Pooled F1:                  {results['pooled_f1']:.4f}")

    # Pooled confusion matrix
    cm = results["pooled_confusion_matrix"]
    print(f"\n--- Pooled Confusion Matrix ---")
    print(f"                  Predicted Healthy  Predicted PD")
    print(f"  Actual Healthy      {cm[0,0]:>5} (TN)        {cm[0,1]:>5} (FP)")
    print(f"  Actual PD           {cm[1,0]:>5} (FN)        {cm[1,1]:>5} (TP)")
    print(f"\n  False negatives (missed PD cases): {cm[1,0]}")
    print(f"  False positives (false alarms):    {cm[0,1]}")

    # Interpretation
    print(f"\n--- Interpretation ---")
    pooled_acc = results["pooled_accuracy"]
    if pooled_acc >= 0.80:
        print(f"  ✅ {pooled_acc:.1%} — strong baseline.")
    elif pooled_acc >= 0.70:
        print(f"  ✅ {pooled_acc:.1%} — solid baseline in expected range (70-80%).")
    elif pooled_acc >= 0.60:
        print(f"  ⚠️  {pooled_acc:.1%} — below expected. Features may need improvement.")
    else:
        print(f"  ❌ {pooled_acc:.1%} — near random. Check feature extraction.")

    print(f"\n  Recall = {results['pooled_recall']:.1%} (of all actual PD cases, how many we caught)")
    if results["pooled_recall"] < 0.70:
        print(f"  ⚠️  Low recall — model is missing many PD cases (false negatives).")
    else:
        print(f"  ✅ Recall is acceptable for a screening aid.")


# ---------------------------------------------------------------------------
# Step 5: Save OOF predictions for later McNemar test
# ---------------------------------------------------------------------------

def save_predictions(results):
    """Save OOF predictions for later comparison with CNN via McNemar test.

    We save:
        - oof_predictions_handcrafted.npy: binary predictions (0 or 1)
        - oof_probabilities_handcrafted.npy: probability scores (0.0 to 1.0)

    These will be compared against the CNN's OOF predictions in Week 2/3.
    """
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/oof_predictions_handcrafted.npy", results["oof_predictions"])
    np.save("data/processed/oof_probabilities_handcrafted.npy", results["oof_probabilities"])
    print(f"\n  Saved OOF predictions to data/processed/oof_predictions_handcrafted.npy")
    print(f"  Saved OOF probabilities to data/processed/oof_probabilities_handcrafted.npy")


# ---------------------------------------------------------------------------
# Step 6: Feature importance (which features matter most?)
# ---------------------------------------------------------------------------

def analyze_feature_importance(features, labels, patient_ids, feature_names):
    """Train logistic regression on ALL data to get feature coefficients.

    NOTE: This is for interpretation only, NOT for evaluation.
    The evaluation uses patient-level CV (above). Here we train on all data
    to get an overall sense of which features matter.

    Args:
        features, labels, patient_ids: full dataset
        feature_names: list of feature names
    """
    print(f"\n--- Feature Importance (trained on all data, for interpretation) ---")

    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Train on all data (NOT for evaluation — just for coefficient inspection)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(features_scaled, labels)

    # Get coefficients (after standardization, so they're comparable)
    coefs = model.coef_[0]

    print(f"  {'Feature':<35} {'Coefficient':<15} {'|Effect|'}")
    print(f"  " + "-" * 65)
    for name, coef in sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True):
        direction = "↑ PD" if coef > 0 else "↓ PD"
        print(f"  {name:<35} {coef:>+10.4f}      {direction}")

    print(f"\n  Interpretation:")
    print(f"  - Positive coefficient = higher feature value → more likely PD")
    print(f"  - Negative coefficient = higher feature value → less likely PD")
    print(f"  - Magnitude (after standardization) = how important the feature is")


# ---------------------------------------------------------------------------
# Main execution block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("Handcrafted Features Baseline — Patient-Level 5-Fold CV")
    print("=" * 70)

    # Step 1: Load the dataset (for labels and patient_ids)
    print("\n1. Loading dataset...")
    images, labels, patient_ids, filenames = load_newhandpd("data/raw/Merged")

    # Step 2: Load the handcrafted features
    print("\n2. Loading handcrafted features...")
    features, feature_names = load_features()

    # Sanity check: features and labels must have same length
    assert len(features) == len(labels), \
        f"Mismatch: {len(features)} features vs {len(labels)} labels"

    # Step 3: Train with patient-level 5-fold CV
    print("\n3. Training logistic regression baseline...")
    results = train_handcrafted_baseline(features, labels, patient_ids, n_splits=5)

    # Step 4: Print results
    print_results(results, feature_names)

    # Step 5: Save OOF predictions for McNemar test
    print("\n4. Saving OOF predictions for later McNemar test...")
    save_predictions(results)

    # Step 6: Feature importance analysis
    print("\n5. Feature importance analysis...")
    analyze_feature_importance(features, labels, patient_ids, feature_names)

    print("\n" + "=" * 70)
    print("BASELINE TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFirst honest accuracy number: {results['pooled_accuracy']:.1%}")
    print(f"  (mean across folds: {results['mean_accuracy']:.1%} ± {results['std_accuracy']:.1%})")
