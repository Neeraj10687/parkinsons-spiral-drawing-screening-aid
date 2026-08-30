"""
Handcrafted features extractor for NewHandPD spiral images.

Extracts 5 geometric features from each spiral image. These features form
our handcrafted baseline, which we compare to the CNN using McNemar's test.

The 5 features:
    1. RADIUS_DEVIATION — how far the spiral deviates from a perfect
       Archimedean spiral fitted to the stroke centroid
    2. STROKE_SMOOTHNESS — mean magnitude of the second derivative of the
       stroke path (higher = jerkier = more likely PD)
    3. STROKE_ENTROPY — entropy of the stroke angle distribution
       (higher = more random directions = more likely PD)
    4. INTERSECTION_COUNT — how many times the spiral crosses itself
       (higher = more chaotic = more likely PD)
    5. MEAN_SQUARED_DISPLACEMENT — mean squared distance of stroke points
       from the centroid (measures spiral spread)

Literature predicts handcrafted features may match or exceed CNN accuracy
on sparse spiral data, because the signal is too thin for CNN feature
maps to learn efficiently.

Methodology reference:
    PMC 2023 — handcrafted features for spiral PD detection
    (specific paper TBD — search "handcrafted features spiral Parkinson")
"""

import os
import sys
import numpy as np
import cv2
from scipy import ndimage
from scipy.stats import entropy as scipy_entropy

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_newhandpd


# ---------------------------------------------------------------------------
# Step 1: Extract the stroke (the black line) from the image
# ---------------------------------------------------------------------------

def extract_stroke(image):
    """Extract the spiral stroke from a grayscale image.

    The image is a 256x256 grayscale array with values in [0, 1].
    The stroke is the dark (low-value) pixels. We threshold and skeletonize
    to get a thin line representing the spiral.

    Args:
        image: numpy array of shape (256, 256, 1) or (256, 256), values in [0, 1]

    Returns:
        binary: numpy array of shape (256, 256), values 0 or 1.
                1 = stroke pixel, 0 = background.
        skeleton: numpy array of shape (256, 256), values 0 or 1.
                  Thinned version of binary (1-pixel-wide line).
    """
    # Remove channel dimension if present: (256, 256, 1) -> (256, 256)
    if image.ndim == 3:
        image = image.squeeze()

    # Convert to uint8 (0-255) for OpenCV operations
    img_uint8 = (image * 255).astype(np.uint8)

    # Threshold: pixels darker than 128 become stroke (1), rest become background (0)
    # The spiral is black on white, so we invert: dark pixels = stroke
    _, binary = cv2.threshold(img_uint8, 128, 1, cv2.THRESH_BINARY_INV)

    # Skeletonize: thin the stroke to a 1-pixel-wide line.
    # Use Guo-Hall thinning algorithm. The constant name differs between
    # OpenCV versions (THINNING_GUO_HALL vs THINNING_GUOHALL), so we handle both.
    thinning_type = getattr(cv2.ximgproc, "THINNING_GUO_HALL",
                            getattr(cv2.ximgproc, "THINNING_GUOHALL", None))
    if thinning_type is None:
        raise AttributeError(
            "cv2.ximgproc has no thinning type constant. "
            "Available: " + str([x for x in dir(cv2.ximgproc) if 'THINNING' in x])
        )
    skeleton = cv2.ximgproc.thinning(
        (binary * 255).astype(np.uint8),
        thinningType=thinning_type
    )
    skeleton = (skeleton > 0).astype(np.uint8)

    return binary, skeleton


# ---------------------------------------------------------------------------
# Step 2: Get the stroke as an ordered path (for path-based features)
# ---------------------------------------------------------------------------

def get_stroke_points(binary):
    """Get all stroke pixel coordinates.

    Args:
        binary: numpy array of shape (256, 256), values 0 or 1

    Returns:
        points: numpy array of shape (N, 2), each row is (row, col) = (y, x)
                Returns None if no stroke pixels found.
    """
    # Find all non-zero pixel coordinates
    # cv2.findNonZero returns shape (N, 1, 2) where each element is (x, y)
    nonzero = cv2.findNonZero(binary.astype(np.uint8))
    if nonzero is None:
        return None

    # Reshape to (N, 2) and convert (x, y) -> (row, col) = (y, x)
    points = nonzero.squeeze().astype(np.float32)  # (N, 2) in (x, y) format
    # Flip to (y, x) for consistency with row/col indexing
    points = points[:, ::-1]  # now (row, col) = (y, x)
    return points


# ---------------------------------------------------------------------------
# Feature 1: Radius Deviation from ideal Archimedean spiral
# ---------------------------------------------------------------------------

def compute_radius_deviation(points):
    """Compute how far the stroke deviates from a perfect spiral.

    A perfect Archimedean spiral has radius growing linearly with angle:
        r = a * theta
    where r is distance from center, theta is angle.

    For each stroke point, we compute:
        1. Distance from centroid (r)
        2. Angle from centroid (theta)
    Then we fit r = a * theta and compute the standard deviation of residuals.

    High deviation = spiral is irregular = more likely PD.
    Low deviation = spiral is smooth and regular = more likely healthy.

    Args:
        points: numpy array of shape (N, 2), each row is (row, col)

    Returns:
        float: standard deviation of (r - a*theta), normalized by mean r.
               Returns 0.0 if not enough points.
    """
    if points is None or len(points) < 10:
        return 0.0

    # Compute centroid
    cy, cx = points.mean(axis=0)

    # Compute radius and angle for each point
    dy = points[:, 0] - cy
    dx = points[:, 1] - cx
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # range [-pi, pi]

    # Unwrap theta to make it monotonically increasing (spiral goes around multiple times)
    # Sort points by angle, then unwrap
    sort_idx = np.argsort(theta)
    theta_sorted = theta[sort_idx]
    r_sorted = r[sort_idx]
    theta_unwrapped = np.unwrap(theta_sorted)

    # Fit r = a * theta (linear regression through origin)
    # a = sum(theta * r) / sum(theta^2)
    if np.sum(theta_unwrapped**2) == 0:
        return 0.0
    a = np.sum(theta_unwrapped * r_sorted) / np.sum(theta_unwrapped**2)

    # Compute residuals (how far each point is from the perfect spiral)
    r_predicted = a * theta_unwrapped
    residuals = r_sorted - r_predicted

    # Return standard deviation of residuals, normalized by mean radius
    mean_r = np.mean(r_sorted)
    if mean_r == 0:
        return 0.0
    return float(np.std(residuals) / mean_r)


# ---------------------------------------------------------------------------
# Feature 2: Stroke Smoothness (mean magnitude of second derivative)
# ---------------------------------------------------------------------------

def compute_stroke_smoothness(points):
    """Compute stroke smoothness via the second derivative.

    The second derivative measures how much the direction is changing.
    A smooth line has low second derivative (direction changes slowly).
    A jerky/trembling line has high second derivative (direction changes rapidly).

    We can't easily order the points along the spiral (that requires path tracing),
    so we approximate by looking at local curvature in a neighborhood.

    Approach: For each point, look at its k nearest neighbors, fit a line,
    and measure how much the point deviates from that line. Average across
    all points.

    High value = jerky strokes = more likely PD.
    Low value = smooth strokes = more likely healthy.

    Args:
        points: numpy array of shape (N, 2)

    Returns:
        float: mean local roughness, normalized by mean pairwise distance.
    """
    if points is None or len(points) < 20:
        return 0.0

    # Sample points to make computation tractable (2000 points max)
    if len(points) > 2000:
        idx = np.random.choice(len(points), 2000, replace=False)
        points = points[idx]

    # Build a KD-tree for fast nearest-neighbor queries
    from scipy.spatial import cKDTree
    tree = cKDTree(points)

    k = 8  # number of neighbors to consider
    roughnesses = []

    for i, point in enumerate(points):
        # Find k nearest neighbors (including the point itself)
        distances, indices = tree.query(point, k=k)
        neighbors = points[indices]

        if len(neighbors) < 3:
            continue

        # Fit a line to the neighbors using PCA
        # The first principal component is the line direction
        centered = neighbors - neighbors.mean(axis=0)
        try:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # The line direction is the eigenvector with the largest eigenvalue
            line_dir = eigenvectors[:, -1]

            # Project each neighbor onto the line, measure perpendicular distance
            projections = centered @ line_dir
            perpendicular = centered - np.outer(projections, line_dir)
            roughness = np.mean(np.linalg.norm(perpendicular, axis=1))

            roughnesses.append(roughness)
        except np.linalg.LinAlgError:
            continue

    if not roughnesses:
        return 0.0

    # Normalize by mean pairwise distance to make scale-invariant
    mean_pairwise_dist = np.mean([np.linalg.norm(points[i] - points[j])
                                   for i in range(0, len(points), 50)
                                   for j in range(i+1, min(i+50, len(points)))])
    if mean_pairwise_dist == 0:
        return 0.0

    return float(np.mean(roughnesses) / mean_pairwise_dist)


# ---------------------------------------------------------------------------
# Feature 3: Entropy of stroke angle distribution
# ---------------------------------------------------------------------------

def compute_stroke_entropy(points, n_bins=36):
    """Compute entropy of the stroke angle distribution.

    For each stroke point, compute the angle from the centroid.
    Then compute the entropy of the angle histogram.

    A smooth spiral has angles spread evenly around the circle -> high entropy.
    Wait, that's wrong. Let me think again.

    Actually: a perfect spiral has angles distributed roughly uniformly around
    the circle (because it goes around multiple times). High entropy = uniform.

    A tremoring spiral has angles clumped in certain directions (because the
    pen gets stuck or tremores in one direction) -> lower entropy.

    Hmm, but that's not quite right either. Let me use a different interpretation:

    We compute the LOCAL angle of the stroke (direction from one point to the
    next), and measure the entropy of these local directions.

    - Smooth spiral: local directions change slowly and predictably -> low entropy
    - Tremoring spiral: local directions are chaotic -> high entropy

    Args:
        points: numpy array of shape (N, 2)
        n_bins: number of bins for the angle histogram (default 36 = 10 degrees each)

    Returns:
        float: entropy of the angle distribution (higher = more random = more likely PD)
    """
    if points is None or len(points) < 20:
        return 0.0

    # Sort points by angle from centroid to get a rough ordering
    cy, cx = points.mean(axis=0)
    dy = points[:, 0] - cy
    dx = points[:, 1] - cx
    angles_from_centroid = np.arctan2(dy, dx)
    sort_idx = np.argsort(angles_from_centroid)
    points_sorted = points[sort_idx]

    # Compute local direction (angle from point i to point i+1)
    deltas = points_sorted[1:] - points_sorted[:-1]
    local_angles = np.arctan2(deltas[:, 0], deltas[:, 1])  # range [-pi, pi]

    # Histogram of local angles
    hist, _ = np.histogram(local_angles, bins=n_bins, range=(-np.pi, np.pi))

    # Normalize to probability distribution
    hist = hist / hist.sum() if hist.sum() > 0 else hist

    # Compute entropy (in nats)
    ent = scipy_entropy(hist, base=np.e)

    # Normalize by maximum possible entropy (log(n_bins))
    max_entropy = np.log(n_bins)
    if max_entropy == 0:
        return 0.0

    return float(ent / max_entropy)


# ---------------------------------------------------------------------------
# Feature 4: Intersection count
# ---------------------------------------------------------------------------

def compute_intersection_count(binary):
    """Count how many times the spiral crosses itself.

    A healthy spiral is smooth and doesn't cross itself much.
    A PD spiral with tremor may cross itself more.

    Approach: We count the number of connected components in the skeleton
    that have more than 2 neighbors (branch points). Each branch point
    roughly corresponds to a self-intersection.

    Args:
        binary: numpy array of shape (256, 256), the thresholded stroke

    Returns:
        int: number of self-intersection points (approximate)
    """
    # Skeletonize using OpenCV thinning (version-tolerant constant lookup)
    thinning_type = getattr(cv2.ximgproc, "THINNING_GUO_HALL",
                            getattr(cv2.ximgproc, "THINNING_GUOHALL", None))
    skeleton = cv2.ximgproc.thinning(
        (binary * 255).astype(np.uint8),
        thinningType=thinning_type
    )
    skeleton = (skeleton > 0).astype(np.uint8)

    # Count neighbors for each skeleton pixel
    # A branch point has 3+ neighbors
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D(skeleton, -1, kernel)

    # Branch points: skeleton pixels with 3+ neighbors
    branch_points = ((skeleton > 0) & (neighbor_count >= 3)).sum()

    return int(branch_points)


# ---------------------------------------------------------------------------
# Feature 5: Mean squared displacement from centroid
# ---------------------------------------------------------------------------

def compute_mean_squared_displacement(points):
    """Compute mean squared distance of stroke points from the centroid.

    This measures how "spread out" the spiral is.
    - A tight, small spiral has low displacement
    - A wide, wandering spiral has high displacement

    Args:
        points: numpy array of shape (N, 2)

    Returns:
        float: mean squared displacement, normalized by image size (256^2)
    """
    if points is None or len(points) < 10:
        return 0.0

    # Compute centroid
    cy, cx = points.mean(axis=0)

    # Compute squared distances
    dy = points[:, 0] - cy
    dx = points[:, 1] - cx
    squared_distances = dx**2 + dy**2

    # Mean squared displacement
    msd = np.mean(squared_distances)

    # Normalize by image size (256^2 = 65536)
    return float(msd / (256**2))


# ---------------------------------------------------------------------------
# Main function: extract all 5 features from one image
# ---------------------------------------------------------------------------

def extract_all_features(image):
    """Extract all 5 handcrafted features from a single image.

    Args:
        image: numpy array of shape (256, 256, 1), values in [0, 1]

    Returns:
        dict with 5 features:
            - radius_deviation
            - stroke_smoothness
            - stroke_entropy
            - intersection_count
            - mean_squared_displacement
    """
    # Step 1: Extract the stroke
    binary, skeleton = extract_stroke(image)

    # Step 2: Get stroke points
    points = get_stroke_points(binary)

    # Step 3: Compute all 5 features
    features = {
        "radius_deviation": compute_radius_deviation(points),
        "stroke_smoothness": compute_stroke_smoothness(points),
        "stroke_entropy": compute_stroke_entropy(points),
        "intersection_count": compute_intersection_count(binary),
        "mean_squared_displacement": compute_mean_squared_displacement(points),
    }

    return features


# ---------------------------------------------------------------------------
# Process the entire dataset
# ---------------------------------------------------------------------------

def extract_features_for_dataset(images):
    """Extract features for all images in the dataset.

    Args:
        images: numpy array of shape (N, 256, 256, 1)

    Returns:
        feature_matrix: numpy array of shape (N, 5), float32
        feature_names: list of 5 feature name strings
    """
    feature_names = [
        "radius_deviation",
        "stroke_smoothness",
        "stroke_entropy",
        "intersection_count",
        "mean_squared_displacement",
    ]

    feature_matrix = []
    for i, img in enumerate(images):
        features = extract_all_features(img)
        feature_vector = [features[name] for name in feature_names]
        feature_matrix.append(feature_vector)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(images)} images")

    feature_matrix = np.array(feature_matrix, dtype=np.float32)
    return feature_matrix, feature_names


# ---------------------------------------------------------------------------
# Main execution block
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Handcrafted Features Extraction")
    print("=" * 60)

    # Load the dataset
    print("\n1. Loading dataset...")
    images, labels, patient_ids, filenames = load_newhandpd("data/raw/Merged")

    # Extract features for a few sample images first (sanity check)
    print("\n2. Sanity check: extracting features for first 5 images...")
    for i in range(5):
        features = extract_all_features(images[i])
        label_str = "Healthy" if labels[i] == 0 else "PD"
        print(f"\n  Image {i+1}: {filenames[i]} ({label_str})")
        for name, value in features.items():
            print(f"    {name:30s}: {value:.4f}")

    # Extract features for all images
    print(f"\n3. Extracting features for all {len(images)} images...")
    feature_matrix, feature_names = extract_features_for_dataset(images)
    print(f"\nFeature matrix shape: {feature_matrix.shape}")
    print(f"Feature names: {feature_names}")

    # Compare healthy vs PD
    print("\n4. Comparing feature values: Healthy vs PD")
    print(f"{'Feature':<30s} {'Healthy (mean±std)':<25s} {'PD (mean±std)':<25s} {'Direction'}")
    print("-" * 100)

    healthy_features = feature_matrix[labels == 0]
    pd_features = feature_matrix[labels == 1]

    for i, name in enumerate(feature_names):
        h_mean, h_std = healthy_features[:, i].mean(), healthy_features[:, i].std()
        p_mean, p_std = pd_features[:, i].mean(), pd_features[:, i].std()
        direction = "PD higher" if p_mean > h_mean else "PD lower"
        print(f"{name:<30s} {h_mean:.4f} ± {h_std:.4f}      {p_mean:.4f} ± {p_std:.4f}      {direction}")

    # Save the feature matrix
    print("\n5. Saving features...")
    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/features_handcrafted.npy", feature_matrix)
    np.save("data/processed/feature_names.npy", np.array(feature_names))
    print("  Saved: data/processed/features_handcrafted.npy")
    print("  Saved: data/processed/feature_names.npy")

    # Correlation matrix
    print("\n6. Feature correlation matrix:")
    corr = np.corrcoef(feature_matrix.T)
    print(f"{'':30s}", end="")
    for name in feature_names:
        print(f"{name[:15]:>16s}", end="")
    print()
    for i, name in enumerate(feature_names):
        print(f"{name:<30s}", end="")
        for j in range(len(feature_names)):
            print(f"{corr[i, j]:>16.2f}", end="")
        print()

    print("\n" + "=" * 60)
    print("Feature extraction complete.")
    print("=" * 60)
    print("\nNext step: Train logistic regression baseline with patient-level CV")
    print("  (src/train.py — the train_handcrafted_baseline function)")
