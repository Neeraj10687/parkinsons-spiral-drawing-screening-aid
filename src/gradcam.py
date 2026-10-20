"""
Grad-CAM plausibility check (not clinical explanation).

Framing: Grad-CAM here is a plausibility check - it verifies the model is
attending to the spiral stroke rather than to image artifacts (corner,
background, watermark). It is NOT a clinical explanation of which stroke
features drive classification, because the sparse-stroke regime makes
Grad-CAM heatmaps difficult to interpret clinically.

To make Grad-CAM substantive, we pair it with a handcrafted-feature
correlation analysis: does Grad-CAM activation within the stroke bounding
box correlate with the radius-deviation feature?

Works for both production models:
    - CNN v4 (spiral_cnn_v4)        - grayscale input (256, 256, 1)
                                       last conv layer: "conv3"
    - MobileNetV2 (spiral_mobilenetv2) - RGB input (256, 256, 3)
                                       last conv layer: "Conv_1" (inside frozen base)

Reference:
    Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization." IEEE ICCV.
    https://arxiv.org/abs/1610.02391

Usage (CLI):
    python src/gradcam.py --model cnn --n-samples 20
    python src/gradcam.py --model mobilenet --n-samples 20
    python src/gradcam.py --model both --n-samples 12
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import cv2
import matplotlib
matplotlib.use("Agg")  # headless backend for CLI
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from src.data_loader import load_newhandpd
from src.features import extract_stroke, extract_features_for_dataset


# ===========================================================================
# Configuration
# ===========================================================================
IMG_SIZE = 256
REPORT_DIR = "reports/gradcam"
PROCESSED_DIR = "data/processed"

# Layer hints - the LAST Conv2D layer we want Grad-CAM to attribute to.
# For MobileNetV2 the relevant conv layer is inside the frozen base; we look
# it up dynamically by walking the model layers, so this hint is just a
# preference list (first match wins).
LAYER_HINTS_CNN = ["conv3", "conv2", "conv1"]
LAYER_HINTS_MOBILENET = ["Conv_1", "out_relu", "block_16_project_BN"]


# ===========================================================================
# Helpers: find last conv layer + RGB conversion
# ===========================================================================

def find_last_conv_layer(model):
    """Find the last Conv2D layer in the model (or its nested base).

    For the CNN v4 the model itself is Sequential and conv3 is the last conv.
    For MobileNetV2 (frozen base + custom head), the last conv layer is
    inside the wrapped MobileNetV2 base model - we descend into it.

    Args:
        model: keras Model

    Returns:
        layer name (string) suitable for Grad-CAM
    """
    # Try preferred hints first - they are stable, well-known names.
    all_sublayers = []
    for layer in model.layers:
        if hasattr(layer, "layers") and len(layer.layers) > 0:
            # Nested model (e.g. MobileNetV2 base)
            all_sublayers.extend(layer.layers)
        else:
            all_sublayers.append(layer)

    hint_pref = (LAYER_HINTS_MOBILENET
                 if "mobilenet" in model.name.lower()
                 else LAYER_HINTS_CNN)
    for hint in hint_pref:
        for layer in all_sublayers:
            if layer.name == hint:
                return layer.name

    # Fallback: pick the last Conv2D layer (by traversal order).
    for layer in reversed(all_sublayers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

    raise RuntimeError(
        f"No Conv2D layer found in model '{model.name}' "
        f"(or its nested base). Cannot compute Grad-CAM."
    )


def get_layer_recursive(model, layer_name):
    """Find a layer by name, descending into nested models if needed.

    Keras' model.get_layer() only searches direct children. For MobileNetV2,
    the conv layer we want (e.g. 'Conv_1') is inside the nested base model,
    so we need to search recursively.

    Args:
        model: keras Model
        layer_name: name of the layer to find

    Returns:
        The Keras layer object.

    Raises:
        ValueError if not found anywhere in the model tree.
    """
    # Check direct children first
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        # If this layer is itself a nested model, descend into it
        if hasattr(layer, "layers") and len(layer.layers) > 0:
            for sublayer in layer.layers:
                if sublayer.name == layer_name:
                    return sublayer
    raise ValueError(
        f"No such layer: {layer_name}. Existing top-level layers: "
        f"{[l.name for l in model.layers]}"
    )


def to_rgb_if_needed(image_batch):
    """Replicate grayscale (n, 256, 256, 1) to (n, 256, 256, 3) if needed.

    Same trick as app/streamlit_app.py and src/train.py.
    """
    if image_batch.shape[-1] == 1:
        return np.repeat(image_batch, 3, axis=-1)
    return image_batch


# ===========================================================================
# 1. Grad-CAM heatmap
# ===========================================================================

def compute_gradcam(model, image, layer_name=None, class_idx=0):
    """Compute Grad-CAM heatmap for a single image.

    Implements the standard Grad-CAM algorithm:
        1. Forward pass to get conv feature maps A (h x w x c) and the
           predicted class score y.
        2. Backprop: gradient of y w.r.t. A, then global-average-pool across
           spatial dims to get per-channel weights alpha_k.
        3. Weighted sum: sum_k alpha_k * A[:,:,k] -> (h, w) raw heatmap.
        4. ReLU (only positive contributions), resize to input size, normalize
           to [0, 1].

    Args:
        model: trained Keras model (CNN v4 or MobileNetV2)
        image: single image batched as (1, 256, 256, C) where C=1 for CNN v4
               and C=3 for MobileNetV2
        layer_name: name of conv layer to attribute to (default: auto-detect)
        class_idx: class index (0 = healthy, 1 = PD; for binary sigmoid we
                   use the positive-class score by default)

    Returns:
        heatmap: np.array (256, 256), values in [0, 1]
        prob:    float, model's predicted P(class_idx)
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)

    # Keras 3 + Sequential model: model.output and layer.output may not be
    # populated (the "never been called" AttributeError). The most robust
    # approach is to NOT build a sub-model and instead use tf.GradientTape
    # directly on the full model, watching the conv layer's output by
    # re-running the forward pass manually and stashing the activation.
    #
    # This works for both:
    #   - Sequential CNN v4 (flat layer list)
    #   - MobileNetV2 (layer 0 is a nested model - we descend into it)

    conv_layer = get_layer_recursive(model, layer_name)

    # --- Forward pass that records the conv activation ---
    # For the simple Sequential CNN v4 case, we walk model.layers in order.
    # For MobileNetV2, the conv layer is inside a nested base - we build a
    # sub-model from the base's input to the conv layer's output.
    if conv_layer in model.layers:
        # CNN v4 case: conv layer is a direct child of the model
        # Forward pass inside the tape so we can watch the activation
        with tf.GradientTape() as tape:
            x = image
            conv_activation = None
            for layer in model.layers:
                x = layer(x, training=False)
                if layer is conv_layer:
                    conv_activation = x
            predictions = x
            # predictions shape is (1, 1) for sigmoid output. reduce_mean
            # collapses it to a scalar safely (handles (1,1), (1,), or scalar).
            p = tf.reduce_mean(predictions)
            score = p if class_idx == 0 else (1.0 - p)
        conv_outputs = conv_activation
    else:
        # MobileNetV2 case: conv layer is inside the nested base model.
        # Find which top-level layer contains this conv layer, then build a
        # sub-model from that base's input to the conv layer's output.
        base = None
        for layer in model.layers:
            if hasattr(layer, "layers") and conv_layer in layer.layers:
                base = layer
                break
        if base is None:
            raise RuntimeError(
                f"Could not find parent base model for conv layer '{layer_name}'. "
                f"This should not happen - check get_layer_recursive()."
            )
        # Calling base() once populates conv_layer.output in Keras 3
        _ = base(image, training=False)
        grad_model = tf.keras.models.Model(
            inputs=base.inputs,
            outputs=[conv_layer.output, base.output],
        )
        with tf.GradientTape() as tape:
            conv_activation, base_output = grad_model(image, training=False)
            x = base_output
            for layer in model.layers:
                if layer is base:
                    continue
                x = layer(x, training=False)
            predictions = x
            p = tf.reduce_mean(predictions)  # (1,1) or (1,) -> scalar
            score = p if class_idx == 0 else (1.0 - p)
        conv_outputs = conv_activation

    # Compute gradients of the class score w.r.t. the conv activation
    grads = tape.gradient(score, conv_outputs)  # (1, h, w, c)
    if grads is None:
        # Should not happen, but guard anyway
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32), float(p)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (c,)

    # conv_outputs shape: (1, h, w, c). Take the first (and only) batch.
    conv_outputs_2d = conv_outputs[0]  # (h, w, c)
    weighted = conv_outputs_2d @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
    weighted = tf.squeeze(weighted)

    heatmap = tf.maximum(weighted, 0.0) / (tf.reduce_max(weighted) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize to input size and normalize to [0, 1]
    heatmap = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    if heatmap.max() > 0:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    return heatmap, float(p)


def compute_gradcam_batch(model, images, layer_name=None, class_idx=0, verbose=True):
    """Compute Grad-CAM heatmaps for a batch of images.

    Args:
        model: trained Keras model
        images: (n, 256, 256, C) array
        layer_name: conv layer name (default: auto-detect)
        class_idx: class index
        verbose: print progress

    Returns:
        heatmaps: (n, 256, 256) array
        probs: (n,) array of predicted P(class_idx)
    """
    if layer_name is None:
        layer_name = find_last_conv_layer(model)
    if verbose:
        print(f"  Using conv layer: {layer_name}")

    n = len(images)
    heatmaps = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    probs = np.zeros(n, dtype=np.float32)

    for i in range(n):
        img = images[i:i + 1]  # (1, 256, 256, C)
        h, p = compute_gradcam(model, img, layer_name=layer_name, class_idx=class_idx)
        heatmaps[i] = h
        probs[i] = p
        if verbose and (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n}")

    return heatmaps, probs


# ===========================================================================
# 2. Stroke-region correlation analysis
# ===========================================================================

def stroke_region_activation(heatmap, image, dilate_px=3):
    """Compute mean Grad-CAM activation inside the stroke region.

    The stroke is the dark line in the image. We binarize the image to find
    stroke pixels, dilate slightly to give the heatmap some tolerance (the
    stroke mask is 1-pixel wide after skeletonization, but the stroke is
    actually several pixels thick), and average the heatmap over those pixels.

    Args:
        heatmap: (256, 256) Grad-CAM heatmap, values in [0, 1]
        image: (256, 256, 1) or (256, 256) original image, values in [0, 1]
        dilate_px: how many pixels to dilate the stroke mask (default 3)

    Returns:
        mean_activation_in_stroke: float
        mean_activation_outside_stroke: float
        stroke_fraction: fraction of image that is stroke (0..1)
    """
    binary, _ = extract_stroke(image)  # (256, 256) 0/1 mask

    # Dilate stroke mask for tolerance
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        binary = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1)

    n_stroke = int(binary.sum())
    if n_stroke == 0:
        # No stroke detected (blank image?) -> use whole image
        return float(heatmap.mean()), float(heatmap.mean()), 1.0

    in_stroke = float(heatmap[binary > 0].mean())
    out_stroke = float(heatmap[binary == 0].mean())
    stroke_fraction = n_stroke / float(IMG_SIZE * IMG_SIZE)
    return in_stroke, out_stroke, stroke_fraction


def gradcam_stroke_correlation(heatmaps, images, handcrafted_features, feature_names):
    """Correlate Grad-CAM stroke-region activation with handcrafted features.

    For each image:
        1. Mask heatmap to the dilated stroke region.
        2. Compute mean Grad-CAM activation within that region.
        3. Compute the "stroke attention ratio":
              in_stroke / (in_stroke + out_stroke)
           A ratio > 0.5 means the model focuses on the stroke more than the
           background, which is what we want.
        4. Correlate this ratio with each handcrafted feature using
           scipy.stats.pearsonr.

    A positive correlation with radius_deviation, for example, suggests the
    CNN is firing on the same signal the handcrafted feature measures.

    Args:
        heatmaps: (n, 256, 256) array of Grad-CAM heatmaps
        images: (n, 256, 256, 1) original images
        handcrafted_features: (n, 5) array
        feature_names: list of 5 feature name strings

    Returns:
        dict with:
            "stroke_activations": (n,) array of mean in-stroke activations
            "stroke_attention_ratio": (n,) array of in/(in+out) ratios
            "correlations": dict {feature_name: (pearson_r, p_value)}
    """
    n = len(heatmaps)
    in_stroke = np.zeros(n, dtype=np.float32)
    out_stroke = np.zeros(n, dtype=np.float32)
    stroke_frac = np.zeros(n, dtype=np.float32)

    print(f"  Computing stroke-region activations for {n} images...")
    for i in range(n):
        h = heatmaps[i]
        img = images[i]
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img
        else:
            img = img[..., np.newaxis] if img.ndim == 2 else img
        s_in, s_out, s_frac = stroke_region_activation(h, img)
        in_stroke[i] = s_in
        out_stroke[i] = s_out
        stroke_frac[i] = s_frac
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{n}")

    eps = 1e-8
    ratio = in_stroke / (in_stroke + out_stroke + eps)

    print(f"\n  Mean stroke attention ratio: {ratio.mean():.4f} "
          f"(>0.5 = model focuses on stroke, not background)")
    print(f"  Mean stroke fraction:         {stroke_frac.mean():.4f}")

    # Correlations
    correlations = {}
    print(f"\n  Correlations (stroke attention ratio vs feature):")
    for fname, fcol in zip(feature_names, handcrafted_features.T):
        valid = np.isfinite(fcol) & np.isfinite(ratio)
        if valid.sum() < 5 or np.std(fcol[valid]) < 1e-8:
            correlations[fname] = (0.0, 1.0)
            continue
        r, p = pearsonr(ratio[valid], fcol[valid])
        correlations[fname] = (float(r), float(p))
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"    {fname:<30} r={r:+.4f}  p={p:.4f}  {sig}")

    return {
        "stroke_activations": in_stroke,
        "background_activations": out_stroke,
        "stroke_attention_ratio": ratio,
        "stroke_fractions": stroke_frac,
        "correlations": correlations,
    }


# ===========================================================================
# 3. Report generation
# ===========================================================================

def overlay_heatmap(image, heatmap, alpha=0.5, cmap="jet"):
    """Overlay a Grad-CAM heatmap on a grayscale image.

    Args:
        image: (256, 256, 1) or (256, 256) image, values in [0, 1]
        heatmap: (256, 256) heatmap, values in [0, 1]
        alpha: heatmap opacity (0..1)
        cmap: matplotlib colormap for heatmap

    Returns:
        rgb: (256, 256, 3) uint8 overlay image
    """
    if image.ndim == 3:
        image = image[..., 0]
    image_uint8 = (image * 255).astype(np.uint8)
    rgb = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2RGB)

    cmap_fn = plt.get_cmap(cmap)
    heat_color = (cmap_fn(heatmap)[..., :3] * 255).astype(np.uint8)
    overlay = cv2.addWeighted(rgb, 1 - alpha, heat_color, alpha, 0)
    return overlay


def generate_gradcam_report(model, model_name, images, labels, predictions, probabilities,
                            handcrafted_features, feature_names, output_dir=REPORT_DIR):
    """Generate a Grad-CAM report for one model.

    Produces:
        1. Sample Grad-CAM visualizations:
           - 4 correct PD, 4 correct healthy, 4 incorrect (any) - 12 panels total
        2. Stroke-region correlation table (printed + saved as .txt)
        3. Scatter plot: stroke attention ratio vs radius_deviation
        4. Side-by-side overlay comparison for a few samples

    Args:
        model: trained Keras model (CNN v4 or MobileNetV2)
        model_name: short name for file labeling ("cnn" or "mobilenet")
        images: (n, 256, 256, 1) grayscale images
        labels: (n,) true labels (0=healthy, 1=PD)
        predictions: (n,) predicted labels
        probabilities: (n,) predicted P(PD)
        handcrafted_features: (n, 5) feature matrix
        feature_names: list of 5 feature names
        output_dir: where to save report files

    Returns:
        dict with paths to all generated artifacts + correlation results
    """
    os.makedirs(output_dir, exist_ok=True)
    n = len(images)
    print(f"\n{'=' * 70}")
    print(f"GENERATING GRAD-CAM REPORT - {model_name.upper()}")
    print(f"{'=' * 70}")
    print(f"  Model:      {model.name}")
    print(f"  Images:     {n}")
    print(f"  Output dir: {output_dir}")

    # --- 1. Convert to RGB if needed + compute all heatmaps ---
    is_mobilenet = "mobilenet" in model.name.lower()
    if is_mobilenet:
        images_for_model = to_rgb_if_needed(images)
    else:
        images_for_model = images

    print(f"\n  Computing Grad-CAM heatmaps for all {n} images...")
    heatmaps, probs = compute_gradcam_batch(model, images_for_model, verbose=True)

    # --- 2. Stroke-region correlation analysis ---
    print(f"\n  --- Stroke-region correlation analysis ---")
    corr_results = gradcam_stroke_correlation(
        heatmaps, images, handcrafted_features, feature_names,
    )

    # Save correlation table as text
    corr_path = os.path.join(output_dir, f"stroke_correlations_{model_name}.txt")
    with open(corr_path, "w") as f:
        f.write(f"Grad-CAM Stroke-Region Correlation - Model: {model.name}\n")
        f.write(f"Dataset: {n} images\n\n")
        f.write(f"Mean stroke attention ratio: "
                f"{corr_results['stroke_attention_ratio'].mean():.4f}\n")
        f.write(f"Mean in-stroke activation:   "
                f"{corr_results['stroke_activations'].mean():.4f}\n")
        f.write(f"Mean background activation:  "
                f"{corr_results['background_activations'].mean():.4f}\n")
        f.write(f"Mean stroke fraction:        "
                f"{corr_results['stroke_fractions'].mean():.4f}\n\n")
        f.write(f"{'Feature':<30} {'Pearson r':<12} {'p-value':<12} {'Sig':<5}\n")
        f.write("-" * 60 + "\n")
        for fname, (r, p) in corr_results["correlations"].items():
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
            f.write(f"{fname:<30} {r:+.4f}      {p:.4f}      {sig}\n")
    print(f"  Saved correlation table -> {corr_path}")

    # --- 3. Scatter plot: stroke attention ratio vs radius_deviation ---
    if "radius_deviation" in feature_names:
        idx = feature_names.index("radius_deviation")
        x = handcrafted_features[:, idx]
        y = corr_results["stroke_attention_ratio"]
        valid = np.isfinite(x) & np.isfinite(y)

        fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
        # Color points by true label
        colors = np.where(labels == 1, "#C0392B", "#27AE60")
        ax.scatter(x[valid], y[valid], c=colors[valid], alpha=0.6, edgecolors="white", s=40)
        if valid.sum() > 5 and np.std(x[valid]) > 1e-8 and np.std(y[valid]) > 1e-8:
            r, p = pearsonr(x[valid], y[valid])
            ax.set_title(
                f"Stroke Attention Ratio vs Radius Deviation\n"
                f"r = {r:+.4f}, p = {p:.4f} (n={valid.sum()})",
                color="#1A5276",
            )
            # Linear fit line
            coef = np.polyfit(x[valid], y[valid], 1)
            xs = np.linspace(x[valid].min(), x[valid].max(), 50)
            ax.plot(xs, np.polyval(coef, xs), "k--", alpha=0.6, linewidth=1.5)
        else:
            ax.set_title("Stroke Attention Ratio vs Radius Deviation", color="#1A5276")
        ax.set_xlabel("radius_deviation (handcrafted feature)", fontsize=11)
        ax.set_ylabel("Stroke attention ratio (Grad-CAM)", fontsize=11)
        ax.grid(True, alpha=0.3)
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", label="PD", markerfacecolor="#C0392B", markersize=8),
            Line2D([0], [0], marker="o", color="w", label="Healthy",
                   markerfacecolor="#27AE60", markersize=8),
        ]
        ax.legend(handles=legend_elements, loc="best")
        scatter_path = os.path.join(output_dir, f"scatter_stroke_vs_radius_{model_name}.png")
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"  Saved scatter plot    -> {scatter_path}")

    # --- 4. Sample Grad-CAM visualizations ---
    correct_pd = np.where((labels == 1) & (predictions == 1))[0]
    correct_h = np.where((labels == 0) & (predictions == 0))[0]
    incorrect = np.where(labels != predictions)[0]

    n_each = 4
    samples = (
        list(correct_pd[:n_each]) +
        list(correct_h[:n_each]) +
        list(incorrect[:n_each])
    )
    titles = (
        ["Correct PD"] * min(n_each, len(correct_pd)) +
        ["Correct Healthy"] * min(n_each, len(correct_h)) +
        ["Incorrect"] * min(n_each, len(incorrect))
    )
    samples = samples[:12]
    titles = titles[:12]

    if len(samples) > 0:
        n_cols = 4
        n_rows = int(np.ceil(len(samples) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5),
                                 constrained_layout=True)
        axes = np.atleast_2d(axes)
        for i, (idx_img, title) in enumerate(zip(samples, titles)):
            r, c = divmod(i, n_cols)
            ax = axes[r, c]
            overlay = overlay_heatmap(images[idx_img], heatmaps[idx_img], alpha=0.55)
            ax.imshow(overlay)
            true_label = "PD" if labels[idx_img] == 1 else "Healthy"
            ax.set_title(
                f"{title}\n"
                f"True: {true_label} | Pred prob: {probabilities[idx_img]:.2f}",
                fontsize=9,
                color="#1A5276",
            )
            ax.axis("off")
        # Hide unused subplots
        for j in range(len(samples), n_rows * n_cols):
            r, c = divmod(j, n_cols)
            axes[r, c].axis("off")
        fig.suptitle(f"Grad-CAM Heatmaps - {model.name}",
                     fontsize=13, color="#1A5276", y=1.0)
        samples_path = os.path.join(output_dir, f"sample_heatmaps_{model_name}.png")
        plt.savefig(samples_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved sample heatmaps -> {samples_path}")

    # --- 5. Save raw heatmaps as .npy for downstream use ---
    heatmaps_path = os.path.join(PROCESSED_DIR, f"gradcam_heatmaps_{model_name}.npy")
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    np.save(heatmaps_path, heatmaps)
    print(f"  Saved raw heatmaps    -> {heatmaps_path}")

    return {
        "output_dir": output_dir,
        "correlations": corr_results["correlations"],
        "stroke_attention_ratio": corr_results["stroke_attention_ratio"],
        "heatmaps": heatmaps,
        "artifacts": {
            "correlation_table": corr_path,
            "scatter_plot": (
                os.path.join(output_dir, f"scatter_stroke_vs_radius_{model_name}.png")
                if "radius_deviation" in feature_names else None
            ),
            "sample_heatmaps": (
                os.path.join(output_dir, f"sample_heatmaps_{model_name}.png")
                if len(samples) > 0 else None
            ),
            "raw_heatmaps": heatmaps_path,
        },
    }


# ===========================================================================
# 4. Model loader (matches Streamlit + train.py architectures)
# ===========================================================================

def load_model_for_gradcam(model_name):
    """Load a trained model + its weights for Grad-CAM analysis.

    Args:
        model_name: "cnn" or "mobilenet"

    Returns:
        (model, weights_loaded: bool)
    """
    from src.model import build_model, build_mobilenetv2

    # Accept multiple weight filename conventions so the script works whether
    # weights were saved by Colab training (cnn_weights.weights.h5) or by
    # the Streamlit cache (cnn_model_weights.weights.h5).
    if model_name == "cnn":
        model = build_model()
        candidates = ["cnn_weights.weights.h5", "cnn_model_weights.weights.h5"]
    elif model_name == "mobilenet":
        model = build_mobilenetv2()
        candidates = ["mobilenet_weights.weights.h5"]
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    weights_path = None
    for c in candidates:
        p = os.path.join(PROCESSED_DIR, c)
        if os.path.exists(p):
            weights_path = p
            break

    if weights_path is not None:
        model.load_weights(weights_path)
        print(f"  Loaded weights: {weights_path}")
        return model, True
    else:
        tried = ", ".join(candidates)
        print(f"  WARNING: no weights file found in {PROCESSED_DIR}/")
        print(f"  Tried: {tried}")
        print(f"  Using untrained model - heatmaps will be random.")
        return model, False


# ===========================================================================
# 5. CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM report for trained models."
    )
    parser.add_argument(
        "--model", choices=["cnn", "mobilenet", "both"], default="both",
        help="Which model to analyze (default: both)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=0,
        help="If >0, only analyze the first N images (default: 0 = all 632)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("GRAD-CAM ANALYSIS - Parkinson's Spiral Drawing Screening Aid")
    print("=" * 70)

    # Load dataset
    print("\nLoading dataset...")
    images, labels, patient_ids, filenames = load_newhandpd("data/raw/Merged")
    print(f"  Loaded {len(images)} images, {len(np.unique(patient_ids))} patients")
    print(f"  PD: {int(labels.sum())}, Healthy: {int(len(labels) - labels.sum())}")

    # Subsample if requested
    if args.n_samples > 0 and args.n_samples < len(images):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(images), size=args.n_samples, replace=False)
        images = images[idx]
        labels = labels[idx]
        patient_ids = patient_ids[idx]
        print(f"  Subsampled to {args.n_samples} images (seed=42)")

    # Load handcrafted features for correlation analysis
    # Use cached features if running on full dataset, otherwise extract fresh
    feat_cache = os.path.join(PROCESSED_DIR, "features_handcrafted.npy")
    names_cache = os.path.join(PROCESSED_DIR, "feature_names.npy")
    if (args.n_samples == 0 and os.path.exists(feat_cache) and os.path.exists(names_cache)):
        print("\nLoading cached handcrafted features...")
        features = np.load(feat_cache)
        feature_names = np.load(names_cache, allow_pickle=True).tolist()
        assert len(features) == len(images), \
            f"Cache size {len(features)} != dataset size {len(images)}. Delete cache and re-run train.py --model baseline."
    else:
        print("\nExtracting handcrafted features (slow, no cache)...")
        features, feature_names = extract_features_for_dataset(images)
    print(f"  Features: {feature_names}")

    # Run predictions for each model (to find correct/incorrect samples)
    models_to_run = []
    if args.model in ["cnn", "both"]:
        models_to_run.append("cnn")
    if args.model in ["mobilenet", "both"]:
        models_to_run.append("mobilenet")

    overall_results = {}

    for mname in models_to_run:
        print(f"\n{'=' * 70}")
        print(f"Loading model: {mname}")
        print(f"{'=' * 70}")
        model, weights_loaded = load_model_for_gradcam(mname)
        if not weights_loaded:
            print(f"  Skipping {mname} - no weights found.")
            continue

        # Predict
        if mname == "mobilenet":
            images_for_pred = to_rgb_if_needed(images)
        else:
            images_for_pred = images
        probs = model.predict(images_for_pred, verbose=0).flatten()
        preds = (probs >= 0.5).astype(int)

        acc = float((preds == labels).mean())
        print(f"  Re-predicted accuracy: {acc:.4f}")

        # Generate report
        result = generate_gradcam_report(
            model=model,
            model_name=mname,
            images=images,
            labels=labels,
            predictions=preds,
            probabilities=probs,
            handcrafted_features=features,
            feature_names=feature_names,
            output_dir=REPORT_DIR,
        )
        overall_results[mname] = result

        # Free memory
        del model
        tf.keras.backend.clear_session()

    # --- Final summary ---
    print("\n" + "=" * 70)
    print("GRAD-CAM ANALYSIS COMPLETE")
    print("=" * 70)
    for mname, res in overall_results.items():
        ratio_mean = float(res["stroke_attention_ratio"].mean())
        print(f"\n  {mname}:")
        print(f"    Stroke attention ratio: {ratio_mean:.4f} (>0.5 = focuses on stroke)")
        print(f"    Artifacts in:           {res['output_dir']}/")
        for k, v in res["artifacts"].items():
            if v:
                print(f"      {k}: {v}")
        print(f"    Correlations (top 3):")
        sorted_corr = sorted(
            res["correlations"].items(),
            key=lambda x: abs(x[1][0]),
            reverse=True,
        )
        for fname, (r, p) in sorted_corr[:3]:
            print(f"      {fname:<30} r={r:+.4f}  p={p:.4f}")


if __name__ == "__main__":
    main()
