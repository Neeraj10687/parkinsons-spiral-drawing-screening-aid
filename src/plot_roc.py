"""
ROC curve plotting for Parkinson's Spiral Drawing Screening Aid.

Generates a publication-quality ROC plot with all models on one graph.
Saves to data/processed/roc_curves_all.png.

Usage:
    python src/plot_roc.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_newhandpd

PROCESSED_DIR = "data/processed"

# Model display config
MODELS = [
    ("lr", "LR Baseline", "#1E8449", "-"),
    ("cnn", "CNN v4", "#2E86C1", "-"),
    ("mobilenet", "MobileNetV2", "#E67E22", "-"),
    ("ensemble", "Ensemble", "#8E44AD", "--"),
]


def main():
    # Load labels
    _, labels, _, _ = load_newhandpd("data/raw/Merged")
    print(f"Labels: {len(labels)} (PD={sum(labels)}, Healthy={len(labels)-sum(labels)})")

    # Setup matplotlib for Chinese font fallback
    import matplotlib.font_manager as fm
    plt.rcParams['font.sans-serif'] = ['Liberation Sans', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6))

    auc_values = {}

    for key, display_name, color, linestyle in MODELS:
        prob_path = os.path.join(PROCESSED_DIR, f"oof_probabilities_{key}.npy")
        if not os.path.exists(prob_path):
            print(f"  Skipping {display_name}: {prob_path} not found")
            continue

        probs = np.load(prob_path)
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = auc(fpr, tpr)
        auc_values[key] = auc_val

        plt.plot(fpr, tpr, color=color, linewidth=2, linestyle=linestyle,
                 label=f"{display_name} (AUC = {auc_val:.3f})")
        print(f"  {display_name}: AUC = {auc_val:.4f}")

    # Diagonal reference
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")

    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves \u2014 Patient-Level 5-Fold CV", fontsize=14, color="#1A5276")
    plt.legend(fontsize=10, loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.tight_layout()

    save_path = os.path.join(PROCESSED_DIR, "roc_curves_all.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.show()

    # Also save individual ROC curves
    for key, display_name, color, _ in MODELS:
        prob_path = os.path.join(PROCESSED_DIR, f"oof_probabilities_{key}.npy")
        if not os.path.exists(prob_path):
            continue

        probs = np.load(prob_path)
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color=color, linewidth=2, label=f"AUC = {auc_val:.3f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
        plt.xlabel("False Positive Rate", fontsize=12)
        plt.ylabel("True Positive Rate", fontsize=12)
        plt.title(f"ROC Curve \u2014 {display_name}", fontsize=14, color="#1A5276")
        plt.legend(fontsize=11, loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        individual_path = os.path.join(PROCESSED_DIR, f"roc_{key}.png")
        plt.savefig(individual_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {individual_path}")


if __name__ == "__main__":
    main()
