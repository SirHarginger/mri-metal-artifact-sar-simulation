"""Protocol comparison and low-SAR parameter sweep utilities."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from src.config import LOW_SAR_PROTOCOL, STANDARD_PROTOCOL
from src.sar_model import compute_relative_sar_map


def protocol_comparison_table(
    sar_summary: dict,
    standard_protocol: dict = STANDARD_PROTOCOL,
    low_sar_protocol: dict = LOW_SAR_PROTOCOL,
) -> pd.DataFrame:
    """Create a compact table comparing the standard and low-SAR protocols."""

    rows = [
        {
            "protocol": standard_protocol["name"],
            "field_strength_t": standard_protocol["field_strength_t"],
            "tr_ms": standard_protocol["tr_ms"],
            "te_ms": standard_protocol["te_ms"],
            "flip_angle_deg": standard_protocol["flip_angle_deg"],
            "refocusing_angle_deg": standard_protocol["refocusing_angle_deg"],
            "bandwidth_khz": standard_protocol["bandwidth_khz"],
            "sequence_factor": standard_protocol["sequence_factor"],
            "max_relative_sar": sar_summary["standard_max_relative_sar"],
            "mean_relative_sar": sar_summary["standard_mean_relative_sar"],
            "sar_reduction_percent": 0.0,
            "comment": "good signal but more SAR/artifact",
        },
        {
            "protocol": low_sar_protocol["name"],
            "field_strength_t": low_sar_protocol["field_strength_t"],
            "tr_ms": low_sar_protocol["tr_ms"],
            "te_ms": low_sar_protocol["te_ms"],
            "flip_angle_deg": low_sar_protocol["flip_angle_deg"],
            "refocusing_angle_deg": low_sar_protocol["refocusing_angle_deg"],
            "bandwidth_khz": low_sar_protocol["bandwidth_khz"],
            "sequence_factor": low_sar_protocol["sequence_factor"],
            "max_relative_sar": sar_summary["low_sar_max_relative_sar"],
            "mean_relative_sar": sar_summary["low_sar_mean_relative_sar"],
            "sar_reduction_percent": sar_summary["mean_sar_reduction_percent"],
            "comment": "safer balance with lower RF burden",
        },
    ]
    return pd.DataFrame(rows)


def _artifact_risk_proxy(protocol: dict) -> float:
    te = float(protocol.get("te_ms", 12.0))
    b0 = float(protocol.get("field_strength_t", 1.5))
    bandwidth = max(float(protocol.get("bandwidth_khz", 200.0)), 1.0)
    return (te / 12.0) * (b0 / 1.5) * (200.0 / bandwidth)


def _snr_preservation_proxy(protocol: dict, reference_protocol: dict) -> float:
    """Estimate relative SNR preservation from protocol parameters.

    This is a simple protocol-level proxy for optimisation ranking. Increasing
    receiver bandwidth reduces SNR, lower flip angles reduce excitation signal,
    and longer TR partly restores signal. It is not a substitute for measured
    image SNR, which is still computed from the DICOM image.
    """

    flip = float(protocol.get("flip_angle_deg", 90.0))
    flip_ref = float(reference_protocol.get("flip_angle_deg", 90.0))
    tr = max(float(protocol.get("tr_ms", 2500.0)), 1.0)
    tr_ref = max(float(reference_protocol.get("tr_ms", 2500.0)), 1.0)
    bandwidth = max(float(protocol.get("bandwidth_khz", 200.0)), 1.0)
    bandwidth_ref = max(float(reference_protocol.get("bandwidth_khz", 200.0)), 1.0)

    flip_term = min(flip / flip_ref, 1.0)
    tr_term = min(np.sqrt(tr / tr_ref), 1.25)
    bandwidth_term = np.sqrt(bandwidth_ref / bandwidth)
    return float(np.clip(flip_term * tr_term * bandwidth_term, 0.0, 1.25))


def run_parameter_sweep(
    e_rms_map: np.ndarray,
    tissue_mask: np.ndarray,
    reference_protocol: dict = STANDARD_PROTOCOL,
) -> pd.DataFrame:
    """Sweep low-SAR protocol parameters and compute relative SAR trends."""

    rows = []
    flip_angles = [60.0, 70.0, 80.0, 90.0]
    refocusing_angles = [110.0, 120.0, 130.0, 150.0, 180.0]
    tr_values = [2500.0, 3500.0, 4200.0, 5000.0]
    te_values = [8.0, 10.0, 12.0, 15.0]
    bandwidth_values = [200.0, 320.0, 500.0]
    sequence_factors = [0.45, 0.55, 0.70, 1.0]

    for flip, refocusing, tr, te, bandwidth, sequence_factor in product(
        flip_angles,
        refocusing_angles,
        tr_values,
        te_values,
        bandwidth_values,
        sequence_factors,
    ):
        protocol = dict(LOW_SAR_PROTOCOL)
        protocol.update(
            {
                "flip_angle_deg": flip,
                "refocusing_angle_deg": refocusing,
                "tr_ms": tr,
                "te_ms": te,
                "bandwidth_khz": bandwidth,
                "sequence_factor": sequence_factor,
            }
        )
        sar_map = compute_relative_sar_map(
            e_rms_map,
            tissue_mask,
            protocol,
            reference_protocol,
        )
        values = sar_map[tissue_mask]
        rows.append(
            {
                "flip_angle_deg": flip,
                "refocusing_angle_deg": refocusing,
                "tr_ms": tr,
                "te_ms": te,
                "bandwidth_khz": bandwidth,
                "sequence_factor": sequence_factor,
                "max_relative_sar": float(values.max()) if values.size else 0.0,
                "mean_relative_sar": float(values.mean()) if values.size else 0.0,
                "artifact_risk_proxy": float(_artifact_risk_proxy(protocol)),
                "snr_preservation_proxy": _snr_preservation_proxy(
                    protocol,
                    reference_protocol,
                ),
            }
        )

    result = pd.DataFrame(rows)
    reference_values = result["mean_relative_sar"]
    if reference_values.size and reference_values.max() > 0:
        result["sar_rank_proxy"] = 1.0 - result["mean_relative_sar"] / reference_values.max()
    else:
        result["sar_rank_proxy"] = 0.0
    result["artifact_control_proxy"] = 1.0 / (1.0 + result["artifact_risk_proxy"])
    result["optimization_score"] = (
        0.45 * result["sar_rank_proxy"]
        + 0.30 * result["snr_preservation_proxy"].clip(0.0, 1.0)
        + 0.25 * result["artifact_control_proxy"]
    )
    return result.sort_values(
        ["optimization_score", "mean_relative_sar", "artifact_risk_proxy"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def select_optimized_protocol(
    sweep_df: pd.DataFrame,
    standard_protocol: dict,
    standard_mean_relative_sar: float,
    min_sar_reduction_percent: float = 60.0,
    min_snr_proxy: float = 0.55,
    max_artifact_risk_proxy: float = 0.75,
) -> tuple[dict, dict]:
    """Select a proposal-aligned low-SAR protocol from the sweep.

    The selected protocol must reduce relative SAR substantially while keeping
    simple protocol-level image-quality proxies within acceptable bounds.
    """

    table = sweep_df.copy()
    if standard_mean_relative_sar > 0:
        table["sar_reduction_percent"] = (
            100.0
            * (standard_mean_relative_sar - table["mean_relative_sar"])
            / standard_mean_relative_sar
        )
    else:
        table["sar_reduction_percent"] = 0.0

    candidates = table[
        (table["sar_reduction_percent"] >= min_sar_reduction_percent)
        & (table["snr_preservation_proxy"] >= min_snr_proxy)
        & (table["artifact_risk_proxy"] <= max_artifact_risk_proxy)
    ]
    if candidates.empty:
        candidates = table[table["sar_reduction_percent"] >= min_sar_reduction_percent]
    if candidates.empty:
        candidates = table

    selected = candidates.sort_values(
        ["optimization_score", "snr_preservation_proxy", "artifact_risk_proxy"],
        ascending=[False, False, True],
    ).iloc[0]

    protocol = dict(STANDARD_PROTOCOL)
    protocol.update(
        {
            "name": "Optimized low-SAR protocol",
            "field_strength_t": float(standard_protocol.get("field_strength_t", 1.5)),
            "flip_angle_deg": float(selected["flip_angle_deg"]),
            "refocusing_angle_deg": float(selected["refocusing_angle_deg"]),
            "tr_ms": float(selected["tr_ms"]),
            "te_ms": float(selected["te_ms"]),
            "bandwidth_khz": float(selected["bandwidth_khz"]),
            "sequence_factor": float(selected["sequence_factor"]),
            "sequence": "Optimized FSE/TSE low-SAR template",
            "parallel_imaging": True,
            "fat_suppression": "STIR preferred",
        }
    )
    summary = {
        "flip_angle_deg": float(selected["flip_angle_deg"]),
        "refocusing_angle_deg": float(selected["refocusing_angle_deg"]),
        "tr_ms": float(selected["tr_ms"]),
        "te_ms": float(selected["te_ms"]),
        "bandwidth_khz": float(selected["bandwidth_khz"]),
        "sequence_factor": float(selected["sequence_factor"]),
        "mean_relative_sar": float(selected["mean_relative_sar"]),
        "max_relative_sar": float(selected["max_relative_sar"]),
        "sar_reduction_percent": float(selected["sar_reduction_percent"]),
        "artifact_risk_proxy": float(selected["artifact_risk_proxy"]),
        "snr_preservation_proxy": float(selected["snr_preservation_proxy"]),
        "optimization_score": float(selected["optimization_score"]),
        "selection_rule": (
            "maximize balanced SAR reduction, SNR preservation, and artifact-control "
            "proxy under low-SAR constraints"
        ),
    }
    return protocol, summary


def optimization_summary_table(
    standard_protocol: dict,
    low_sar_protocol: dict,
    optimized_protocol: dict,
    standard_mean_relative_sar: float,
    low_sar_mean_relative_sar: float,
    optimized_summary: dict,
) -> pd.DataFrame:
    """Create a proposal-focused protocol optimisation summary."""

    def reduction(mean_value: float) -> float:
        if standard_mean_relative_sar <= 0:
            return 0.0
        return 100.0 * (standard_mean_relative_sar - mean_value) / standard_mean_relative_sar

    rows = [
        {
            "protocol": "Standard clinical template",
            "flip_angle_deg": standard_protocol["flip_angle_deg"],
            "refocusing_angle_deg": standard_protocol["refocusing_angle_deg"],
            "tr_ms": standard_protocol["tr_ms"],
            "te_ms": standard_protocol["te_ms"],
            "bandwidth_khz": standard_protocol["bandwidth_khz"],
            "sequence_factor": standard_protocol["sequence_factor"],
            "mean_relative_sar": standard_mean_relative_sar,
            "sar_reduction_percent": 0.0,
            "snr_preservation_proxy": 1.0,
            "artifact_risk_proxy": _artifact_risk_proxy(standard_protocol),
            "role": "baseline comparison",
        },
        {
            "protocol": "Low-SAR template",
            "flip_angle_deg": low_sar_protocol["flip_angle_deg"],
            "refocusing_angle_deg": low_sar_protocol["refocusing_angle_deg"],
            "tr_ms": low_sar_protocol["tr_ms"],
            "te_ms": low_sar_protocol["te_ms"],
            "bandwidth_khz": low_sar_protocol["bandwidth_khz"],
            "sequence_factor": low_sar_protocol["sequence_factor"],
            "mean_relative_sar": low_sar_mean_relative_sar,
            "sar_reduction_percent": reduction(low_sar_mean_relative_sar),
            "snr_preservation_proxy": _snr_preservation_proxy(
                low_sar_protocol,
                standard_protocol,
            ),
            "artifact_risk_proxy": _artifact_risk_proxy(low_sar_protocol),
            "role": "predefined safety template",
        },
        {
            "protocol": "Optimized low-SAR protocol",
            "flip_angle_deg": optimized_protocol["flip_angle_deg"],
            "refocusing_angle_deg": optimized_protocol["refocusing_angle_deg"],
            "tr_ms": optimized_protocol["tr_ms"],
            "te_ms": optimized_protocol["te_ms"],
            "bandwidth_khz": optimized_protocol["bandwidth_khz"],
            "sequence_factor": optimized_protocol["sequence_factor"],
            "mean_relative_sar": optimized_summary["mean_relative_sar"],
            "sar_reduction_percent": optimized_summary["sar_reduction_percent"],
            "snr_preservation_proxy": optimized_summary["snr_preservation_proxy"],
            "artifact_risk_proxy": optimized_summary["artifact_risk_proxy"],
            "role": "selected balance of SAR and image quality",
        },
    ]
    return pd.DataFrame(rows)
