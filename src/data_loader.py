"""
Data loader for NewHandPD spiral dataset.

Loads images, extracts patient IDs from filenames, verifies no patient
appears in both classes.

Folder structure (after unzipping NewHandPD):
    data/raw/HealthySpiral/HealthySpiral/sp1-H1.jpg, sp1-H2.jpg, ...
    data/raw/PatientSpiral/PatientSpiral/sp1-P1.jpg, sp1-P2.jpg, ...

Filename format: sp{spiral_number}-{patient_id}.jpg
    - sp1, sp2, sp3, sp4 = which of the 4 spirals per patient
    - H1..H35 = healthy patient IDs (label 0)
    - P1..P31 = Parkinson's patient IDs (label 1)

Methodology reference:
    (2025) "Finger drawing on smartphone screens enables early Parkinson's
    disease detection..." PMC12258557.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC12258557
"""

import os
import re
import numpy as np
import cv2
from PIL import Image


# --- Configuration ---
IMAGE_SIZE = 256  # target size (width = height = 256)


def extract_patient_id(filename):
    """Extract patient ID from a NewHandPD filename.

    Example: "sp1-H10.jpg" -> "H10"
             "sp4-P15.jpg" -> "P15"

    The patient ID is everything between the hyphen and the .jpg extension.
    The first letter (H or P) tells us the class.

    Args:
        filename: image filename (e.g., "sp1-H10.jpg") or full path

    Returns:
        str: patient ID (e.g., "H10", "P15")
    """
    # Get just the filename (in case a full path was passed)
    basename = os.path.basename(filename)
    # Remove the .jpg extension
    name_without_ext = os.path.splitext(basename)[0]  # "sp1-H10"
    # Split on the hyphen and take the second part
    parts = name_without_ext.split("-")
    if len(parts) != 2:
        raise ValueError(f"Unexpected filename format: {filename}")
    patient_id = parts[1]  # "H10"
    return patient_id


def extract_class(patient_id):
    """Determine the class label from the patient ID.

    Handles both original IDs (H1, P1) and prefixed merged IDs (NH1, OP1).
    Looks for 'H' or 'P' anywhere in the string.

    Args:
        patient_id: e.g., "H10", "P15", "NH10", or "OP15"

    Returns:
        int: 0 for healthy, 1 for Parkinson's
    """
    upper_id = patient_id.upper()
    if "H" in upper_id:
        return 0  # healthy
    elif "P" in upper_id:
        return 1  # Parkinson's
    else:
        raise ValueError(f"Unexpected patient ID format: {patient_id}")
#####################################
def load_and_preprocess_image(image_path, target_size=IMAGE_SIZE):
    """Load a single image and preprocess it for the CNN.

    Steps:
        1. Read the image in grayscale (1 channel, not 3)
        2. Resize to 256x256 (some NewHandPD images may be different sizes)
        3. Normalize pixel values from [0, 255] to [0.0, 1.0]
        4. Reshape to (256, 256, 1) — the extra dimension is the channel axis

    Args:
        image_path: full path to the .jpg file
        target_size: image will be resized to (target_size, target_size)

    Returns:
        np.array of shape (target_size, target_size, 1), dtype float32, values in [0, 1]
    """
    # Read as grayscale. cv2.IMREAD_GRAYSCALE = 0.
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Resize to target_size x target_size.
    # INTER_AREA is the recommended interpolation for downscaling.
    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Normalize to [0, 1] by dividing by 255.
    # Convert to float32 first to avoid integer division.
    img = img.astype(np.float32) / 255.0

    # Add the channel axis: (256, 256) -> (256, 256, 1)
    # CNNs expect (height, width, channels), even for grayscale.
    img = np.expand_dims(img, axis=-1)

    return img


def find_image_folder(data_dir, class_name):
    """Find the folder containing images for a given class.

    NewHandPD extracts differently depending on the OS and tool:
    - Sometimes: data/raw/HealthySpiral/HealthySpiral/sp1-H1.jpg (nested)
    - Sometimes: data/raw/HealthySpiral/sp1-H1.jpg (flat)
    - Sometimes: data/raw/healthy/sp1-H1.jpg (lowercase)

    This function auto-detects the actual folder by looking for one that
    contains .jpg files.

    Args:
        data_dir: path to the folder containing the class subfolders
        class_name: e.g., "HealthySpiral" or "PatientSpiral"

    Returns:
        str: full path to the folder containing .jpg files

    Raises:
        FileNotFoundError if no folder with .jpg files is found
    """
    # Try several possible folder structures, in order of likelihood
    candidates = [
        os.path.join(data_dir, class_name, class_name),  # nested: HealthySpiral/HealthySpiral/
        os.path.join(data_dir, class_name),              # flat: HealthySpiral/
        os.path.join(data_dir, class_name.lower(), class_name.lower()),  # lowercase nested
        os.path.join(data_dir, class_name.lower()),      # lowercase flat
    ]

    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        # Check if this folder (or any subfolder, excluding __MACOSX) has .jpg files
        for root, dirs, files in os.walk(candidate):
            if "__MACOSX" in root:
                continue
            if any(f.lower().endswith(".jpg") for f in files):
                return candidate

    # If we get here, none of the candidates worked
    raise FileNotFoundError(
        f"Could not find {class_name} image folder. Tried: {candidates}. "
        f"Run 'find data/raw -type d' to see your actual folder structure."
    )


def find_all_images(folders):
    """Find all .jpg files in the given folders.

    Walks each folder recursively and collects all .jpg files.
    Skips __MACOSX folders (junk from macOS zip extraction).

    Args:
        folders: list of folder paths to search

    Returns:
        list of full paths to .jpg files
    """
    all_images = []
    for folder in folders:
        for root, dirs, files in os.walk(folder):
            # Skip __MACOSX junk folders
            if "__MACOSX" in root:
                continue
            for filename in files:
                # Only keep .jpg files (case-insensitive)
                if filename.lower().endswith(".jpg"):
                    all_images.append(os.path.join(root, filename))
    return all_images


def load_newhandpd(data_dir="data/raw"):
    """Load NewHandPD spiral images and labels.

    Expected structure:
        data_dir/HealthySpiral/HealthySpiral/sp1-H1.jpg, ...
        data_dir/PatientSpiral/PatientSpiral/sp1-P1.jpg, ...

    Args:
        data_dir: path to the folder containing HealthySpiral/ and PatientSpiral/

    Returns:
        images: np.array of shape (n_samples, 256, 256, 1), float32, [0,1]
        labels: np.array of shape (n_samples,), int32 — 0=healthy, 1=PD
        patient_ids: np.array of shape (n_samples,), dtype='<U5' (strings) — e.g., "H10"
        filenames: list of original filenames (for debugging)
    """
    # Auto-detect the folder structure (nested vs flat, case variations)
    healthy_folder = find_image_folder(data_dir, "HealthySpiral")
    patient_folder = find_image_folder(data_dir, "PatientSpiral")
    print(f"Healthy folder: {healthy_folder}")
    print(f"Patient folder: {patient_folder}")

    # Find all .jpg files in both folders
    image_paths = find_all_images([healthy_folder, patient_folder])
    print(f"Found {len(image_paths)} images total")

    # Load each image and extract metadata
    images = []
    labels = []
    patient_ids = []
    filenames = []

    for path in image_paths:
        # Extract patient ID and class from filename
        patient_id = extract_patient_id(path)
        label = extract_class(patient_id)

        # Load and preprocess the image
        img = load_and_preprocess_image(path)

        images.append(img)
        labels.append(label)
        patient_ids.append(patient_id)
        filenames.append(os.path.basename(path))

    # Convert lists to numpy arrays
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    patient_ids = np.array(patient_ids)

    print(f"Loaded {len(images)} images")
    print(f"  Healthy (label 0): {np.sum(labels == 0)} images")
    print(f"  Parkinson's (label 1): {np.sum(labels == 1)} images")
    print(f"  Unique patients: {len(np.unique(patient_ids))}")

    return images, labels, patient_ids, filenames


def verify_no_class_leakage(patient_ids, labels):
    """Verify that no patient appears in both classes.

    This catches a critical bug: if patient "P1" somehow had a label 0 in one
    image and label 1 in another, our patient-level CV would still leak.

    Args:
        patient_ids: array of patient IDs
        labels: array of labels (0 or 1)

    Raises:
        AssertionError if any patient has mixed labels
    """
    unique_patients = np.unique(patient_ids)
    for patient in unique_patients:
        patient_labels = labels[patient_ids == patient]
        unique_labels = np.unique(patient_labels)
        assert len(unique_labels) == 1, (
            f"Patient {patient} has mixed labels: {unique_labels}. "
            f"This should never happen — check your data loading."
        )
    print(f"Verification passed: all {len(unique_patients)} patients have consistent labels.")


def patient_level_split(images, labels, patient_ids, n_splits=5, random_state=42):
    """StratifiedGroupKFold split ensuring no patient overlap.

    Uses sklearn.model_selection.StratifiedGroupKFold to ensure:
    1. Each patient appears in only one fold (no leakage)
    2. Class balance is preserved across folds

    Args:
        images, labels, patient_ids: loaded data
        n_splits: number of CV folds (default 5)
        random_state: for reproducibility

    Returns:
        list of (train_idx, test_idx) tuples, length n_splits
    """
    from sklearn.model_selection import StratifiedGroupKFold

    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(sgk.split(images, labels, groups=patient_ids))
    return splits


def verify_no_patient_overlap(patient_ids, train_idx, test_idx):
    """Assert that no patient appears in both train and test.

    This is the most important check in the whole project.
    If this fails, the entire evaluation is invalid.

    Args:
        patient_ids: array of patient identifiers
        train_idx, test_idx: index arrays

    Raises:
        AssertionError if any patient appears in both splits
    """
    train_patients = set(patient_ids[train_idx])
    test_patients = set(patient_ids[test_idx])
    overlap = train_patients & test_patients
    assert len(overlap) == 0, (
        f"DATA LEAKAGE DETECTED: {len(overlap)} patients appear in both "
        f"train and test: {sorted(overlap)[:5]}"
    )


# ---------------------------------------------------------------------------
# Main execution block — run this file directly to test the loader
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Load the data
    images, labels, patient_ids, filenames = load_newhandpd("data/raw/Merged")

    # Verify no patient has mixed labels
    verify_no_class_leakage(patient_ids, labels)

    # Print some stats
    print("\n--- Dataset Stats ---")
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Patient IDs shape: {patient_ids.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Unique patients: {len(np.unique(patient_ids))}")

    # Show 5 sample filenames with their extracted patient IDs and labels
    print("\n--- Sample (first 5) ---")
    for i in range(5):
        print(f"  {filenames[i]} -> patient_id={patient_ids[i]}, label={labels[i]}")

    # Test the patient-level split
    print("\n--- Testing 5-fold patient-level split ---")
    splits = patient_level_split(images, labels, patient_ids, n_splits=5)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Verify no patient overlap
        verify_no_patient_overlap(patient_ids, train_idx, test_idx)

        train_patients = set(patient_ids[train_idx])
        test_patients = set(patient_ids[test_idx])
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        print(f"  Fold {fold_idx+1}: "
              f"train={len(train_idx)} images ({len(train_patients)} patients, "
              f"{np.sum(train_labels==1)} PD), "
              f"test={len(test_idx)} images ({len(test_patients)} patients, "
              f"{np.sum(test_labels==1)} PD) — NO LEAKAGE")

    print("\nAll checks passed. Data loader is ready.")
