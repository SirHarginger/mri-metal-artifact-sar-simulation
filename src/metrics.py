"""Image quality and relative SAR metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from skimage import filters, morphology

from src.preprocessing import pixel_area_mm2


def _safe_values(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)[mask.astype(bool)]
    return values[np.isfinite(values)]


def compute_snr(image: np.ndarray, tissue_mask: np.ndarray, noise_mask: np.ndarray) -> float:
    """Compute simple SNR as mean tissue signal divided by noise SD."""

    signal = _safe_values(image, tissue_mask)
    noise = _safe_values(image, noise_mask)
    if signal.size == 0:
        return 0.0
    noise_sd = float(np.std(noise)) if noise.size else float(np.std(image))
    if noise_sd <= 1e-8:
        noise_sd = 1e-8
    return float(np.mean(signal) / noise_sd)


def compute_cnr(
    image: np.ndarray,
    tissue_mask: np.ndarray,
    reference_mask: np.ndarray,
    noise_mask: np.ndarray,
) -> float:
    """Compute simple CNR between tissue and reference regions."""

    tissue = _safe_values(image, tissue_mask)
    reference = _safe_values(image, reference_mask)
    noise = _safe_values(image, noise_mask)
    if tissue.size == 0 or reference.size == 0:
        return 0.0
    noise_sd = float(np.std(noise)) if noise.size else float(np.std(image))
    if noise_sd <= 1e-8:
        noise_sd = 1e-8
    return float(abs(np.mean(tissue) - np.mean(reference)) / noise_sd)


def compute_sharpness(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Estimate sharpness using mean Sobel gradient magnitude."""

    gradient = filters.sobel(np.asarray(image, dtype=np.float32))
    if mask is not None and mask.any():
        gradient = gradient[mask.astype(bool)]
    return float(np.mean(gradient)) if gradient.size else 0.0


def compute_artifact_mask(
    reference_image: np.ndarray,
    test_image: np.ndarray,
    body_mask: np.ndarray,
    threshold: float = 0.12,
) -> np.ndarray:
    """Identify image regions substantially changed by artefact simulation."""

    difference = np.abs(np.asarray(reference_image) - np.asarray(test_image))
    candidate = (difference > threshold) & body_mask
    candidate = morphology.closing(candidate, morphology.disk(2))
    candidate = morphology.remove_small_objects(candidate, max_size=15)
    return candidate.astype(bool)


def artifact_area_mm2(
    artifact_mask: np.ndarray,
    pixel_spacing_mm: list[float] | tuple[float, float] | None,
) -> float:
    """Return artefact area in square millimeters."""

    return float(artifact_mask.sum() * pixel_area_mm2(pixel_spacing_mm))


def mean_signal_near_implant(image: np.ndarray, near_implant_mask: np.ndarray) -> float:
    """Mean image signal near the implant."""

    values = _safe_values(image, near_implant_mask)
    return float(np.mean(values)) if values.size else 0.0


def signal_loss_percentage(
    reference_image: np.ndarray,
    test_image: np.ndarray,
    near_implant_mask: np.ndarray,
) -> float:
    """Compute percentage signal loss near implant relative to reference."""

    reference = mean_signal_near_implant(reference_image, near_implant_mask)
    test = mean_signal_near_implant(test_image, near_implant_mask)
    if reference <= 1e-8:
        return 0.0
    return float(100.0 * (reference - test) / reference)


def image_quality_row(
    label: str,
    image: np.ndarray,
    reference_image: np.ndarray,
    masks: dict[str, np.ndarray],
    near_implant_mask: np.ndarray,
    sar_map: np.ndarray,
    pixel_spacing_mm: list[float] | tuple[float, float] | None,
    comment: str,
) -> dict:
    """Build one row of image quality and SAR metrics."""

    artifact_mask = compute_artifact_mask(reference_image, image, masks["body"])
    tissue_values = sar_map[masks["tissue"]]
    reference_mask = masks["implant"] if masks["implant"].any() else masks["background"]
    return {
        "Protocol": label,
        "Relative SAR": float(np.mean(tissue_values)) if tissue_values.size else 0.0,
        "Max relative SAR": float(np.max(tissue_values)) if tissue_values.size else 0.0,
        "SNR": compute_snr(image, masks["tissue"], masks["background"]),
        "CNR": compute_cnr(image, masks["tissue"], reference_mask, masks["background"]),
        "Artefact area": artifact_area_mm2(artifact_mask, pixel_spacing_mm),
        "Sharpness": compute_sharpness(image, masks["body"]),
        "Mean signal near implant": mean_signal_near_implant(image, near_implant_mask),
        "Signal loss percentage": signal_loss_percentage(
            reference_image,
            image,
            near_implant_mask,
        ),
        "Comment": comment,
    }


def build_metrics_summary(
    reference_image: np.ndarray,
    standard_artifact_image: np.ndarray,
    low_sar_artifact_image: np.ndarray,
    optimized_artifact_image: np.ndarray,
    corrected_image: np.ndarray,
    masks: dict[str, np.ndarray],
    near_implant_mask: np.ndarray,
    standard_sar_map: np.ndarray,
    low_sar_map: np.ndarray,
    optimized_sar_map: np.ndarray,
    pixel_spacing_mm: list[float] | tuple[float, float] | None,
) -> pd.DataFrame:
    """Create the metrics CSV table requested by the project brief."""

    rows = [
        image_quality_row(
            "Standard protocol",
            reference_image,
            reference_image,
            masks,
            near_implant_mask,
            standard_sar_map,
            pixel_spacing_mm,
            "good signal but more SAR/artifact",
        ),
        image_quality_row(
            "Low-SAR protocol",
            low_sar_artifact_image,
            reference_image,
            masks,
            near_implant_mask,
            low_sar_map,
            pixel_spacing_mm,
            "safer balance",
        ),
        image_quality_row(
            "Artefact-corrupted",
            standard_artifact_image,
            reference_image,
            masks,
            near_implant_mask,
            standard_sar_map,
            pixel_spacing_mm,
            "off-resonance distortion and signal loss",
        ),
        image_quality_row(
            "Optimized low-SAR protocol",
            optimized_artifact_image,
            reference_image,
            masks,
            near_implant_mask,
            optimized_sar_map,
            pixel_spacing_mm,
            "selected low-SAR protocol balancing SAR and image quality",
        ),
        image_quality_row(
            "Artifact-corrected optimized protocol",
            corrected_image,
            reference_image,
            masks,
            near_implant_mask,
            optimized_sar_map,
            pixel_spacing_mm,
            "artifact-corrected using inverse simulation",
        ),
    ]
    table = pd.DataFrame(rows)
    standard_sar = float(table.loc[table["Protocol"] == "Standard protocol", "Relative SAR"].iloc[0])
    low_sar_area = float(table.loc[table["Protocol"] == "Low-SAR protocol", "Artefact area"].iloc[0])
    optimized_area = float(
        table.loc[table["Protocol"] == "Optimized low-SAR protocol", "Artefact area"].iloc[0]
    )
    table["SAR reduction vs standard (%)"] = 0.0
    if standard_sar > 1e-12:
        table["SAR reduction vs standard (%)"] = (
            100.0 * (standard_sar - table["Relative SAR"].astype(float)) / standard_sar
        )
    table["Artefact reduction vs low-SAR (%)"] = 0.0
    if low_sar_area > 1e-8:
        corrected_index = table["Protocol"] == "Artifact-corrected optimized protocol"
        corrected_area = table.loc[corrected_index, "Artefact area"].astype(float)
        table.loc[corrected_index, "Artefact reduction vs low-SAR (%)"] = (
            100.0 * (low_sar_area - corrected_area) / low_sar_area
        )
    table["Artefact reduction vs optimized (%)"] = 0.0
    if optimized_area > 1e-8:
        corrected_index = table["Protocol"] == "Artifact-corrected optimized protocol"
        corrected_area = table.loc[corrected_index, "Artefact area"].astype(float)
        table.loc[corrected_index, "Artefact reduction vs optimized (%)"] = (
            100.0 * (optimized_area - corrected_area) / optimized_area
        )
    table["Image quality preservation score"] = (
        0.35 * (table["SNR"].astype(float) / max(float(table["SNR"].max()), 1e-8))
        + 0.25 * (table["CNR"].astype(float) / max(float(table["CNR"].max()), 1e-8))
        + 0.20 * (table["Sharpness"].astype(float) / max(float(table["Sharpness"].max()), 1e-8))
        + 0.20 * (1.0 - table["Signal loss percentage"].clip(lower=0.0).astype(float) / 100.0)
    )
    return table
