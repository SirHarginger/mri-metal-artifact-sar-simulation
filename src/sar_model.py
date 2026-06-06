"""Relative SAR-like model for comparative MRI protocol analysis."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from src.config import (
    LOW_SAR_PROTOCOL,
    REFERENCE_E_RMS_V_PER_M,
    STANDARD_PROTOCOL,
    TISSUE_CONDUCTIVITY_S_PER_M,
    TISSUE_DENSITY_KG_PER_M3,
)


def estimate_e_rms_map(
    shape: tuple[int, int],
    implant_mask: np.ndarray,
    tissue_mask: np.ndarray,
    base_e_rms: float = REFERENCE_E_RMS_V_PER_M,
) -> np.ndarray:
    """Estimate a relative RMS electric-field map.

    The field is spatially smooth with a mild implant-edge enhancement. It is
    intended for comparative protocol trends, not patient-specific dosimetry.
    """

    rows, cols = shape
    yy, xx = np.indices(shape, dtype=np.float32)
    x_gradient = 0.9 + 0.2 * (xx / max(cols - 1, 1))
    y_gradient = 0.95 + 0.1 * (yy / max(rows - 1, 1))

    if implant_mask.any():
        distance = ndi.distance_transform_edt(~implant_mask)
        edge_enhancement = 1.0 + 0.65 * np.exp(-distance / 8.0)
    else:
        edge_enhancement = np.ones(shape, dtype=np.float32)

    e_field = float(base_e_rms) * x_gradient * y_gradient * edge_enhancement
    e_field = np.where(tissue_mask, e_field, 0.0)
    return e_field.astype(np.float32)


def protocol_scaling(
    protocol: dict,
    reference_protocol: dict = STANDARD_PROTOCOL,
) -> float:
    """Calculate RF protocol scaling for the relative SAR-like model."""

    alpha = float(protocol.get("flip_angle_deg", 90.0))
    alpha_ref = float(reference_protocol.get("flip_angle_deg", 90.0))
    beta = float(protocol.get("refocusing_angle_deg", 180.0))
    beta_ref = float(reference_protocol.get("refocusing_angle_deg", 180.0))
    tr = max(float(protocol.get("tr_ms", 2500.0)), 1.0)
    tr_ref = max(float(reference_protocol.get("tr_ms", 2500.0)), 1.0)
    sequence_factor = float(protocol.get("sequence_factor", 1.0))
    b0 = float(protocol.get("field_strength_t", 1.5))
    b0_ref = float(reference_protocol.get("field_strength_t", 1.5))

    return (
        (alpha / alpha_ref) ** 2
        * (tr_ref / tr)
        * (beta / beta_ref) ** 2
        * sequence_factor
        * (b0 / b0_ref) ** 2
    )


def compute_relative_sar_map(
    e_rms_map: np.ndarray,
    tissue_mask: np.ndarray,
    protocol: dict,
    reference_protocol: dict = STANDARD_PROTOCOL,
    sigma: float = TISSUE_CONDUCTIVITY_S_PER_M,
    rho: float = TISSUE_DENSITY_KG_PER_M3,
) -> np.ndarray:
    """Compute a relative SAR-like map using sigma * |E_rms|^2 / rho.

    The result is comparative and model-based. It is not a scanner SAR readout,
    not a local SAR validation, and not a clinical safety certification metric.
    """

    base_sar = float(sigma) * np.square(np.asarray(e_rms_map, dtype=np.float32)) / float(rho)
    sar_map = base_sar * protocol_scaling(protocol, reference_protocol)
    return np.where(tissue_mask, sar_map, 0.0).astype(np.float32)


def summarize_sar_maps(
    standard_sar_map: np.ndarray,
    low_sar_map: np.ndarray,
    tissue_mask: np.ndarray,
) -> dict:
    """Summarize standard and low-SAR maps."""

    standard_values = standard_sar_map[tissue_mask]
    low_values = low_sar_map[tissue_mask]
    standard_mean = float(np.mean(standard_values)) if standard_values.size else 0.0
    low_mean = float(np.mean(low_values)) if low_values.size else 0.0
    standard_max = float(np.max(standard_values)) if standard_values.size else 0.0
    low_max = float(np.max(low_values)) if low_values.size else 0.0

    mean_reduction = 0.0
    max_reduction = 0.0
    if standard_mean > 0:
        mean_reduction = 100.0 * (standard_mean - low_mean) / standard_mean
    if standard_max > 0:
        max_reduction = 100.0 * (standard_max - low_max) / standard_max

    return {
        "standard_max_relative_sar": standard_max,
        "standard_mean_relative_sar": standard_mean,
        "low_sar_max_relative_sar": low_max,
        "low_sar_mean_relative_sar": low_mean,
        "mean_sar_reduction_percent": float(mean_reduction),
        "max_sar_reduction_percent": float(max_reduction),
    }


def default_protocols() -> tuple[dict, dict]:
    """Return copies of the default standard and low-SAR protocols."""

    return dict(STANDARD_PROTOCOL), dict(LOW_SAR_PROTOCOL)
