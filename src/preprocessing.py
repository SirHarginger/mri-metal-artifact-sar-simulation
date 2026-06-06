"""Image preprocessing utilities."""

from __future__ import annotations

import numpy as np


def normalize_intensity(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
) -> np.ndarray:
    """Normalize image intensity to a robust 0-1 range."""

    data = np.asarray(image, dtype=np.float32)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.zeros_like(data, dtype=np.float32)

    lower, upper = np.percentile(finite, [lower_percentile, upper_percentile])
    if upper <= lower:
        lower = float(np.min(finite))
        upper = float(np.max(finite))
    if upper <= lower:
        return np.zeros_like(data, dtype=np.float32)

    normalized = (data - lower) / (upper - lower)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32)


def pixel_area_mm2(pixel_spacing_mm: list[float] | tuple[float, float] | None) -> float:
    """Return pixel area in square millimeters, falling back to 1 mm2."""

    if pixel_spacing_mm is None or len(pixel_spacing_mm) < 2:
        return 1.0
    try:
        return float(pixel_spacing_mm[0]) * float(pixel_spacing_mm[1])
    except (TypeError, ValueError):
        return 1.0


def border_noise_mask(shape: tuple[int, int], border_fraction: float = 0.08) -> np.ndarray:
    """Create a conservative border mask for noise estimation fallbacks."""

    rows, cols = shape
    border = max(1, int(min(rows, cols) * border_fraction))
    mask = np.zeros(shape, dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    return mask
