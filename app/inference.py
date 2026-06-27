"""
inference.py — HTAN segmentation inference core.

Pure inference logic operating on an already-loaded model. Model loading and
the per-modality registry live in `app.services` so the FastAPI process loads
weights once at startup and reuses them across requests.

Locked decisions (see Medilink_Handoff_-_HTAN_2_N2_Results.md):
    * TTA averages RAW LOGITS, then applies sigmoid ONCE. This matches the
      averaging done in HTAN/utils/trainer.py. Averaging probabilities
      (sigmoid-per-aug, then mean) is NOT equivalent and is not used.
    * No `confidence_score` is reported. The model is not calibrated, so a
      mean-probability "confidence" would be misleading. Detection is binary;
      everything else is relative to the image.
    * No physical measurements (mm, cm²). Pixel spacing is unknown without
      DICOM metadata, so only %-of-image, pixel spans, counts and location
      are reported.

The model's forward() returns raw logits (HTAN_2/HTAN_1 end in a Conv2d with
no activation), which is why sigmoid is applied here, after averaging.
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from scipy import ndimage

IMG_SIZE = 256

# Matches the training normalization used across isic/glas/bowl datasets.
NORM_MEAN = [0.5, 0.5, 0.5]
NORM_STD = [0.5, 0.5, 0.5]

# Components smaller than this (in pixels, at 256x256) are dropped as noise.
MIN_COMPONENT_PX = 50

_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def preprocess(image: Image.Image, device: str) -> tuple[torch.Tensor, tuple[int, int]]:
    """PIL image -> (1, 3, 256, 256) float tensor on `device`, plus original (W, H)."""
    original_size = image.size
    tensor = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(device)
    return tensor, original_size


# ---------------------------------------------------------------------------
# Test-time augmentation — average LOGITS, sigmoid once
# ---------------------------------------------------------------------------
def _logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(x)


def predict_basic_tta(model, img: torch.Tensor) -> np.ndarray:
    """3 views: original + h-flip + v-flip. Fast; recommended for the API."""
    with torch.no_grad():
        acc = _logits(model, img)
        acc = acc + TF.hflip(_logits(model, TF.hflip(img)))
        acc = acc + TF.vflip(_logits(model, TF.vflip(img)))
        prob = torch.sigmoid(acc / 3.0)
    return prob.squeeze().float().cpu().numpy()


def predict_full_tta(model, img: torch.Tensor) -> np.ndarray:
    """8 views: original + 3 flips + 3 rotations + (rot90 + h-flip). ~2.5x slower."""
    with torch.no_grad():
        logits = [
            _logits(model, img),
            TF.hflip(_logits(model, TF.hflip(img))),
            TF.vflip(_logits(model, TF.vflip(img))),
            TF.hflip(TF.vflip(_logits(model, TF.vflip(TF.hflip(img))))),
            TF.rotate(_logits(model, TF.rotate(img, 90)), -90),
            TF.rotate(_logits(model, TF.rotate(img, 180)), -180),
            TF.rotate(_logits(model, TF.rotate(img, 270)), -270),
            TF.rotate(TF.hflip(_logits(model, TF.hflip(TF.rotate(img, 90)))), -90),
        ]
        prob = torch.sigmoid(torch.stack(logits).mean(0))
    return prob.squeeze().float().cpu().numpy()


# ---------------------------------------------------------------------------
# Mask post-processing
# ---------------------------------------------------------------------------
def postprocess_mask(prob_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold -> drop tiny components -> fill internal holes. Returns uint8 {0,1}."""
    raw = (prob_map > threshold).astype(np.uint8)

    labeled, num = ndimage.label(raw)
    cleaned = np.zeros_like(raw)
    for i in range(1, num + 1):
        component = labeled == i
        if component.sum() >= MIN_COMPONENT_PX:
            cleaned[component] = 1

    return ndimage.binary_fill_holes(cleaned).astype(np.uint8)


# ---------------------------------------------------------------------------
# Descriptors
# ---------------------------------------------------------------------------
def describe_location(mask: np.ndarray) -> str:
    """Human-readable region of the mask centroid (e.g. 'upper-left', 'center')."""
    h, w = mask.shape
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return "none"

    cy, cx = ys.mean() / h, xs.mean() / w
    v = "upper" if cy < 0.33 else ("lower" if cy > 0.66 else "central")
    hz = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")

    if hz == "center" and v == "central":
        return "center"
    if hz == "center":
        return v
    if v == "central":
        return hz
    return f"{v}-{hz}"


def relative_size_label(area_percent: float) -> str:
    if area_percent == 0:
        return "none"
    if area_percent < 5:
        return "small"
    if area_percent < 20:
        return "medium"
    return "large"


def estimate_severity(area_percent: float) -> str:
    """
    Coarse screening priority from segmented area only.
    Relative bucket, NOT a clinical grade or diagnosis.
    """
    if area_percent == 0:
        return "none"
    if area_percent < 5:
        return "mild"
    if area_percent < 20:
        return "moderate"
    if area_percent < 40:
        return "severe"
    return "critical"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def segment(
    image: Image.Image,
    model,
    device: str,
    model_name: str,
    modality: str,
    tta_mode: str = "basic",
    include_raw: bool = False,
) -> dict:
    """
    Run segmentation on a PIL image with a pre-loaded model.

    Returns a JSON-serializable dict. `probability_map` / `binary_mask` are
    only included when include_raw=True (kept out of the API response by default).
    """
    img_tensor, original_size = preprocess(image, device)

    prob_map = (
        predict_full_tta(model, img_tensor)
        if tta_mode == "full"
        else predict_basic_tta(model, img_tensor)
    )

    binary_mask = postprocess_mask(prob_map)
    labeled, num_components = ndimage.label(binary_mask)

    total_px = IMG_SIZE * IMG_SIZE
    segmented_px = int(binary_mask.sum())
    segmented_area_pct = round(segmented_px / total_px * 100, 2)

    components = []
    for i in range(1, num_components + 1):
        component = labeled == i
        comp_px = int(component.sum())
        ys, xs = np.where(component)
        span = max(
            int(ys.max() - ys.min()) if len(ys) else 0,
            int(xs.max() - xs.min()) if len(xs) else 0,
        )
        components.append(
            {
                "component_id": i,
                "area_px": comp_px,
                "area_percent": round(comp_px / total_px * 100, 2),
                "largest_span_px": span,
                "location": describe_location(component.astype(np.uint8)),
            }
        )
    components.sort(key=lambda c: c["area_px"], reverse=True)

    result = {
        "target_detected": bool(segmented_px > 0),
        "segmented_area_px": segmented_px,
        "segmented_area_percent": segmented_area_pct,
        "relative_size": relative_size_label(segmented_area_pct),
        "num_components": num_components,
        "largest_component_span_px": components[0]["largest_span_px"] if components else 0,
        "location": describe_location(binary_mask),
        "component_details": components,
        "severity_estimate": estimate_severity(segmented_area_pct),
        "model": model_name,
        "modality": modality,
        "tta_mode": tta_mode,
        "image_size": list(original_size),
        "note": (
            "Segmentation/screening output only. No physical measurements (mm, cm²) "
            "are reported because pixel spacing is unknown without DICOM metadata. "
            "All metrics are relative to image dimensions. Not a clinical diagnosis — "
            "confirm with a qualified professional."
        ),
    }

    if include_raw:
        result["probability_map"] = prob_map.tolist()
        result["binary_mask"] = binary_mask.tolist()

    return result
