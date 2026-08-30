# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Exploratory Data Analysis — NewHandPD Spiral Dataset
#
# **Project:** Parkinson's Spiral Drawing Screening Aid
# **Authors:** Neeraj N (25MCA042) & Roopak M (25MCA048)
#
# This notebook explores the NewHandPD dataset before we build any models.
# We look at:
# 1. Class distribution (healthy vs Parkinson's)
# 2. Images per patient
# 3. Sample images from each class
# 4. Original image sizes (before our 256×256 resize)
# 5. Pixel intensity distribution
#
# **Why EDA matters:** Looking at your data before modeling prevents 80% of bugs.
# If we see something weird here (e.g., all PD images are much smaller than healthy),
# we fix it NOW, not after the CNN fails to train.

# %%
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter

# Add project root to path so we can import from src/
# (This lets us run the notebook from the project root)
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import (
    load_newhandpd,
    find_image_folder,
    find_all_images,
    extract_patient_id,
    extract_class,
)

# Set up matplotlib for clean plots
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 10

# Create figures directory if it doesn't exist
os.makedirs("reports/figures", exist_ok=True)

print("Libraries loaded. Ready for EDA.")

# %% [markdown]
# ## 1. Load the dataset
#
# We use our verified data loader from `src/data_loader.py`.

# %%
images, labels, patient_ids, filenames = load_newhandpd("data/raw/Merged")

print(f"\nDataset loaded:")
print(f"  Images shape: {images.shape}")
print(f"  Labels shape: {labels.shape}")
print(f"  Patient IDs shape: {patient_ids.shape}")

# %% [markdown]
# ## 2. Class distribution
#
# How many healthy vs Parkinson's images do we have?
# A balanced dataset is easier to train; an imbalanced one needs class weighting.

# %%
class_names = ["Healthy (0)", "Parkinson's (1)"]
class_counts = [np.sum(labels == 0), np.sum(labels == 1)]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(class_names, class_counts, color=["#2ecc71", "#e74c3c"], edgecolor="black")
ax.set_ylabel("Number of images")
ax.set_title("Class Distribution")
ax.set_ylim(0, max(class_counts) * 1.15)

# Add count labels on top of bars
for bar, count in zip(bars, class_counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
            str(count), ha="center", va="bottom", fontweight="bold")

plt.tight_layout()
plt.savefig("reports/figures/class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nClass distribution:")
print(f"  Healthy:     {class_counts[0]} images ({class_counts[0]/len(labels)*100:.1f}%)")
print(f"  Parkinson's: {class_counts[1]} images ({class_counts[1]/len(labels)*100:.1f}%)")
print(f"  Ratio:       {class_counts[0]/class_counts[1]:.2f} : 1 (healthy : PD)")
print(f"\n  Interpretation: {'Well balanced' if 0.8 < class_counts[0]/class_counts[1] < 1.2 else 'Slightly imbalanced — consider class weighting'}")

# %% [markdown]
# ## 3. Images per patient
#
# How many spiral images does each patient have?
# NewHandPD says each patient drew 4 spirals, but some may have more or fewer.
# This matters for understanding the effective sample size.

# %%
patient_counts = Counter(patient_ids)
counts = list(patient_counts.values())

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(counts, bins=range(1, max(counts) + 2), edgecolor="black", color="#3498db", align="left")
ax.set_xlabel("Number of images per patient")
ax.set_ylabel("Number of patients")
ax.set_title("Images per Patient Distribution")
ax.set_xticks(range(1, max(counts) + 1))
plt.tight_layout()
plt.savefig("reports/figures/images_per_patient.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nImages per patient:")
print(f"  Min:    {min(counts)} images")
print(f"  Max:    {max(counts)} images")
print(f"  Mean:   {np.mean(counts):.1f} images")
print(f"  Median: {np.median(counts):.1f} images")
print(f"  Total patients: {len(counts)}")
print(f"  Total images:   {sum(counts)}")

# %% [markdown]
# ## 4. Sample images from each class
#
# Let's look at actual spirals. This is the most important EDA step —
# you need to see what your data looks like.
#
# **What to look for:**
# - Healthy spirals should look smoother, more regular
# - PD spirals may look shakier, more irregular
# - If they look identical to you, the CNN will struggle too

# %%
def show_sample_images(images, labels, filenames, n_samples=5):
    """Show n_samples images from each class, side by side."""
    fig, axes = plt.subplots(2, n_samples, figsize=(15, 6))

    # Healthy samples (label 0)
    healthy_idx = np.where(labels == 0)[0]
    healthy_samples = np.random.choice(healthy_idx, size=min(n_samples, len(healthy_idx)), replace=False)

    for i, idx in enumerate(healthy_samples):
        ax = axes[0, i]
        # Remove the channel dimension for display: (256,256,1) -> (256,256)
        ax.imshow(images[idx].squeeze(), cmap="gray")
        ax.set_title(filenames[idx], fontsize=9)
        ax.axis("off")
    axes[0, 0].set_ylabel("Healthy", fontsize=12, fontweight="bold")

    # PD samples (label 1)
    pd_idx = np.where(labels == 1)[0]
    pd_samples = np.random.choice(pd_idx, size=min(n_samples, len(pd_idx)), replace=False)

    for i, idx in enumerate(pd_samples):
        ax = axes[1, i]
        ax.imshow(images[idx].squeeze(), cmap="gray")
        ax.set_title(filenames[idx], fontsize=9)
        ax.axis("off")
    axes[1, 0].set_ylabel("Parkinson's", fontsize=12, fontweight="bold")

    plt.suptitle("Sample Spiral Images from Each Class", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("reports/figures/sample_images.png", dpi=150, bbox_inches="tight")
    plt.show()

# Set random seed for reproducible samples
np.random.seed(42)
show_sample_images(images, labels, filenames, n_samples=5)

# %% [markdown]
# ## 5. Original image sizes (before resize)
#
# Our data loader resizes everything to 256×256. But what were the original sizes?
# If they vary a lot, some images lost more information than others during resizing.
# This is just informational — we can't change it, but we should know.

# %%
# Reload images WITHOUT resizing to check original sizes
healthy_folder = find_image_folder("data/raw/Merged", "HealthySpiral")
patient_folder = find_image_folder("data/raw/Merged", "PatientSpiral")
all_image_paths = find_all_images([healthy_folder, patient_folder])

original_sizes = []
for path in all_image_paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        original_sizes.append(img.shape[:2])  # (height, width)

original_sizes = np.array(original_sizes)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Height distribution
axes[0].hist(original_sizes[:, 0], bins=30, edgecolor="black", color="#9b59b6")
axes[0].set_xlabel("Height (pixels)")
axes[0].set_ylabel("Count")
axes[0].set_title("Original Image Heights")
axes[0].axvline(x=256, color="red", linestyle="--", label="Our resize target (256)")
axes[0].legend()

# Width distribution
axes[1].hist(original_sizes[:, 1], bins=30, edgecolor="black", color="#1abc9c")
axes[1].set_xlabel("Width (pixels)")
axes[1].set_ylabel("Count")
axes[1].set_title("Original Image Widths")
axes[1].axvline(x=256, color="red", linestyle="--", label="Our resize target (256)")
axes[1].legend()

plt.tight_layout()
plt.savefig("reports/figures/original_image_sizes.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nOriginal image sizes:")
print(f"  Height: min={original_sizes[:,0].min()}, max={original_sizes[:,0].max()}, mean={original_sizes[:,0].mean():.0f}")
print(f"  Width:  min={original_sizes[:,1].min()}, max={original_sizes[:,1].max()}, mean={original_sizes[:,1].mean():.0f}")
print(f"  Unique size combinations: {len(np.unique(original_sizes, axis=0))}")
print(f"\n  Note: All images are resized to 256x256 for the CNN. The resize is necessary")
print(f"  because the original sizes vary significantly.")

# %% [markdown]
# ## 6. Pixel intensity distribution
#
# After normalization, pixel values are in [0, 1].
# Most pixels should be near 1.0 (white background), with a smaller spike near 0.0 (black strokes).
# If the distribution looks weird (e.g., all gray), something's wrong with preprocessing.

# %%
# Flatten all images into one array of pixel values
all_pixels = images.flatten()

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(all_pixels, bins=50, edgecolor="black", color="#34495e")
ax.set_xlabel("Pixel value (normalized)")
ax.set_ylabel("Count")
ax.set_title("Pixel Intensity Distribution (after normalization)")
ax.set_yscale("log")  # log scale because white pixels dominate
ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="0.5 threshold")
ax.legend()

plt.tight_layout()
plt.savefig("reports/figures/pixel_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nPixel intensity stats (normalized [0, 1]):")
print(f"  Min:    {all_pixels.min():.4f}")
print(f"  Max:    {all_pixels.max():.4f}")
print(f"  Mean:   {all_pixels.mean():.4f}")
print(f"  Median: {np.median(all_pixels):.4f}")
print(f"  Std:    {all_pixels.std():.4f}")

# Count white vs black pixels
white_fraction = np.mean(all_pixels > 0.9)
black_fraction = np.mean(all_pixels < 0.1)
print(f"\n  White pixels (>0.9): {white_fraction*100:.1f}% — this is the background")
print(f"  Black pixels (<0.1): {black_fraction*100:.1f}% — this is the spiral stroke")
print(f"  Gray pixels (0.1-0.9): {(1-white_fraction-black_fraction)*100:.1f}% — anti-aliasing")

# %% [markdown]
# ## 7. Patient-level statistics
#
# How many unique patients in each class?
# This is the number that matters for statistical power.

# %%
healthy_patients = set(patient_ids[labels == 0])
pd_patients = set(patient_ids[labels == 1])

print(f"Patient statistics:")
print(f"  Unique healthy patients: {len(healthy_patients)}")
print(f"  Unique PD patients:      {len(pd_patients)}")
print(f"  Total unique patients:   {len(healthy_patients) + len(pd_patients)}")

# Check for any overlap (should be zero — verified in data_loader, but double-check)
overlap = healthy_patients & pd_patients
print(f"  Patients in both classes: {len(overlap)} (should be 0)")

# Images per patient, by class
healthy_images_per_patient = [patient_counts[p] for p in healthy_patients]
pd_images_per_patient = [patient_counts[p] for p in pd_patients]

print(f"\nImages per patient by class:")
print(f"  Healthy: mean={np.mean(healthy_images_per_patient):.1f}, min={min(healthy_images_per_patient)}, max={max(healthy_images_per_patient)}")
print(f"  PD:      mean={np.mean(pd_images_per_patient):.1f}, min={min(pd_images_per_patient)}, max={max(pd_images_per_patient)}")

# %% [markdown]
# ## 8. Cross-validation fold distribution
#
# Let's visualize how the 5-fold patient-level split distributes patients and images.
# Each fold should have roughly the same number of patients and the same class balance.

# %%
from src.data_loader import patient_level_split, verify_no_patient_overlap

splits = patient_level_split(images, labels, patient_ids, n_splits=5, random_state=42)

fold_stats = []
for fold_idx, (train_idx, test_idx) in enumerate(splits):
    verify_no_patient_overlap(patient_ids, train_idx, test_idx)

    train_patients = len(set(patient_ids[train_idx]))
    test_patients = len(set(patient_ids[test_idx]))
    train_pd = np.sum(labels[train_idx] == 1)
    test_pd = np.sum(labels[test_idx] == 1)
    train_total = len(train_idx)
    test_total = len(test_idx)

    fold_stats.append({
        "fold": fold_idx + 1,
        "train_patients": train_patients,
        "test_patients": test_patients,
        "train_images": train_total,
        "test_images": test_total,
        "train_pd": train_pd,
        "test_pd": test_pd,
        "test_pd_ratio": test_pd / test_total,
    })

# Print table
print(f"{'Fold':<6} {'Train Pat':<11} {'Test Pat':<10} {'Train Img':<11} {'Test Img':<10} {'Test PD':<9} {'PD Ratio':<10}")
print("-" * 67)
for s in fold_stats:
    print(f"{s['fold']:<6} {s['train_patients']:<11} {s['test_patients']:<10} {s['train_images']:<11} {s['test_images']:<10} {s['test_pd']:<9} {s['test_pd_ratio']:.2%}")

print(f"\n  All folds verified: ZERO patient overlap between train and test.")

# Visualize fold sizes
fig, ax = plt.subplots(figsize=(8, 4))
fold_nums = [s["fold"] for s in fold_stats]
test_sizes = [s["test_images"] for s in fold_stats]
test_pds = [s["test_pd"] for s in fold_stats]
test_healthys = [s["test_images"] - s["test_pd"] for s in fold_stats]

x = np.arange(len(fold_nums))
width = 0.35
ax.bar(x - width/2, test_healthys, width, label="Healthy (test)", color="#2ecc71")
ax.bar(x + width/2, test_pds, width, label="PD (test)", color="#e74c3c")
ax.set_xlabel("Fold")
ax.set_ylabel("Number of test images")
ax.set_title("Test Set Composition per Fold")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {f}" for f in fold_nums])
ax.legend()

plt.tight_layout()
plt.savefig("reports/figures/fold_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Summary
#
# Key findings from the EDA:

# %%
print("=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"""
Dataset: NewHandPD (Spiral drawings)
Total images: {len(images)}
Image shape: {images.shape[1:]} (grayscale, 256x256, normalized)

Class distribution:
  Healthy:     {class_counts[0]} images ({class_counts[0]/len(labels)*100:.1f}%)
  Parkinson's: {class_counts[1]} images ({class_counts[1]/len(labels)*100:.1f}%)
  Balance:     {'Good' if 0.8 < class_counts[0]/class_counts[1] < 1.2 else 'Slightly imbalanced'}

Patient statistics:
  Unique patients: {len(healthy_patients) + len(pd_patients)}
  Healthy: {len(healthy_patients)} patients
  PD: {len(pd_patients)} patients
  Avg images per patient: {np.mean(counts):.1f}

Cross-validation:
  5-fold StratifiedGroupKFold
  All folds verified: ZERO patient leakage
  Test fold sizes: {[s['test_images'] for s in fold_stats]} images

Key observations:
  - Dataset is {'well-balanced' if 0.8 < class_counts[0]/class_counts[1] < 1.2 else 'slightly imbalanced'}
  - Original image sizes vary — resize to 256x256 is necessary
  - Pixel distribution is bimodal (white background + black strokes)
  - Patient-level CV splits are clean — no leakage

Next steps:
  1. Handcrafted features extraction (src/features.py)
  2. Logistic regression baseline with patient-level CV
  3. CNN training (Week 2)
""")

print("Figures saved to: reports/figures/")
print("  - class_distribution.png")
print("  - images_per_patient.png")
print("  - sample_images.png")
print("  - original_image_sizes.png")
print("  - pixel_distribution.png")
print("  - fold_distribution.png")
print("\nEDA complete. Ready to move to feature extraction.")
