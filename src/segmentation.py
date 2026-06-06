"""Implant/signal-void segmentation for real MRI DICOM images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, io as skio, measure, morphology

from src.config import SEGMENTATION
from src.preprocessing import border_noise_mask


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask)
    if labels.max() == 0:
        return np.zeros_like(mask, dtype=bool)
    regions = measure.regionprops(labels)
    largest = max(regions, key=lambda region: region.area)
    return labels == largest.label


def load_manual_mask(
    mask_dir: str | Path,
    dicom_path: str | Path,
    target_shape: tuple[int, int],
) -> np.ndarray | None:
    """Load an optional manual mask matching the DICOM stem.

    Supported formats are NPY and common image files. Any non-zero value is
    treated as part of the implant/signal-void mask.
    """

    directory = Path(mask_dir)
    if not directory.exists():
        return None

    stem = Path(dicom_path).stem
    candidates = []
    for suffix in (".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidates.extend(directory.glob(f"{stem}*{suffix}"))
        candidates.extend(directory.glob(f"*mask*{suffix}"))

    for path in sorted(set(candidates)):
        if path.suffix.lower() == ".npy":
            mask = np.load(path)
        else:
            mask = skio.imread(path)
        if mask.ndim > 2:
            mask = mask[..., 0]
        if mask.shape != target_shape:
            continue
        return np.asarray(mask) > 0
    return None


def estimate_body_mask(normalized_image: np.ndarray) -> np.ndarray:
    """Estimate the body/tissue support while excluding empty background."""

    image = np.asarray(normalized_image, dtype=np.float32)
    smoothed = ndi.gaussian_filter(image, sigma=1.2)
    try:
        threshold = filters.threshold_otsu(smoothed)
    except ValueError:
        threshold = SEGMENTATION["body_threshold_floor"]

    threshold = max(float(threshold) * 0.45, SEGMENTATION["body_threshold_floor"])
    body = smoothed > threshold
    body = morphology.remove_small_objects(body, max_size=127)
    body = morphology.closing(body, morphology.disk(4))
    body = ndi.binary_fill_holes(body)
    body = _largest_component(body)

    if body.sum() < image.size * 0.05:
        body = image > np.percentile(image, 25)
        body = morphology.remove_small_objects(body, max_size=127)
        body = ndi.binary_fill_holes(body)
        body = _largest_component(body)

    return body.astype(bool)


def automatic_signal_void_mask(
    normalized_image: np.ndarray,
    body_mask: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Segment dark implant/signal-void candidates inside the body mask."""

    image = np.asarray(normalized_image, dtype=np.float32)
    body_values = image[body_mask]
    if body_values.size == 0:
        return np.zeros_like(image, dtype=bool), "automatic-empty-body"

    low_threshold = np.percentile(body_values, SEGMENTATION["low_signal_percentile"])
    candidates = (image <= low_threshold) & body_mask
    candidates = morphology.opening(candidates, morphology.disk(1))
    candidates = morphology.closing(candidates, morphology.disk(2))
    candidates = morphology.remove_small_objects(
        candidates,
        max_size=SEGMENTATION["min_object_size"] - 1,
    )

    if candidates.any():
        candidates = _largest_component(candidates)

    min_pixels = max(8, int(image.size * SEGMENTATION["min_implant_fraction"]))
    method = "automatic-low-signal-threshold"
    if candidates.sum() < min_pixels:
        dark_score = ndi.gaussian_filter(1.0 - image, sigma=2.0)
        dark_score = np.where(body_mask, dark_score, -np.inf)
        center = np.unravel_index(np.argmax(dark_score), image.shape)
        radius = max(3, int(min(image.shape) * 0.025))
        yy, xx = np.indices(image.shape)
        candidates = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2
        candidates &= body_mask
        method = "automatic-dark-region-fallback"

    candidates = morphology.dilation(
        candidates,
        morphology.disk(SEGMENTATION["implant_dilation_px"]),
    )
    candidates &= body_mask
    return candidates.astype(bool), method


def segment_masks(
    normalized_image: np.ndarray,
    manual_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Generate implant/signal-void, tissue, and background/noise masks."""

    body_mask = estimate_body_mask(normalized_image)

    if manual_mask is not None:
        implant_mask = np.asarray(manual_mask, dtype=bool) & body_mask
        method = "manual-mask"
    else:
        implant_mask, method = automatic_signal_void_mask(normalized_image, body_mask)

    tissue_mask = body_mask & ~implant_mask
    background_mask = ~body_mask
    if background_mask.sum() < normalized_image.size * 0.02:
        background_mask = border_noise_mask(normalized_image.shape)

    masks = {
        "body": body_mask.astype(bool),
        "implant": implant_mask.astype(bool),
        "tissue": tissue_mask.astype(bool),
        "background": background_mask.astype(bool),
    }
    info = {
        "method": method,
        "implant_pixels": int(implant_mask.sum()),
        "tissue_pixels": int(tissue_mask.sum()),
        "background_pixels": int(background_mask.sum()),
    }
    return masks, info


def near_implant_ring(
    implant_mask: np.ndarray,
    tissue_mask: np.ndarray,
    radius_px: int = 8,
) -> np.ndarray:
    """Return tissue pixels near the implant/signal-void mask."""

    if not implant_mask.any():
        return np.zeros_like(implant_mask, dtype=bool)
    expanded = morphology.dilation(implant_mask, morphology.disk(radius_px))
    return expanded & tissue_mask & ~implant_mask
