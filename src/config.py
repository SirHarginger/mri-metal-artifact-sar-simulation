"""Project configuration and default protocol templates."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DICOM_FILENAME = "MR.0962_25.Image 2.0004.dcm"
DICOM_DIR = PROJECT_ROOT / "data" / "dicom"
DICOM_PATH = DICOM_DIR / DICOM_FILENAME
MASK_DIR = PROJECT_ROOT / "data" / "masks"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MAPS_DIR = OUTPUT_DIR / "maps"
METRICS_DIR = OUTPUT_DIR / "metrics"
REPORTS_DIR = OUTPUT_DIR / "reports"

OUTPUT_FOLDERS = {
    "figures": FIGURES_DIR,
    "maps": MAPS_DIR,
    "metrics": METRICS_DIR,
    "reports": REPORTS_DIR,
}

GYROMAGNETIC_RATIO_HZ_PER_T = 42.57747892e6

TISSUE_CONDUCTIVITY_S_PER_M = 0.6
TISSUE_DENSITY_KG_PER_M3 = 1000.0
REFERENCE_E_RMS_V_PER_M = 1.0

SUSCEPTIBILITY = {
    "tissue_ppm": 0.0,
    "titanium_implant_ppm": 180.0,
}

STANDARD_PROTOCOL = {
    "name": "Standard protocol",
    "field_strength_t": 1.5,
    "tr_ms": 2500.0,
    "te_ms": 12.0,
    "flip_angle_deg": 90.0,
    "refocusing_angle_deg": 180.0,
    "bandwidth_khz": 200.0,
    "sequence": "FSE/TSE",
    "sequence_factor": 1.0,
    "parallel_imaging": False,
    "fat_suppression": "scanner/default",
}

LOW_SAR_PROTOCOL = {
    "name": "Low-SAR protocol",
    "field_strength_t": 1.5,
    "tr_ms": 4200.0,
    "te_ms": 10.0,
    "flip_angle_deg": 70.0,
    "refocusing_angle_deg": 120.0,
    "bandwidth_khz": 320.0,
    "sequence": "FSE/TSE; SEMAC/MAVRIC-inspired correction if available",
    "sequence_factor": 0.55,
    "parallel_imaging": True,
    "fat_suppression": "STIR preferred",
}

SEGMENTATION = {
    "body_threshold_floor": 0.04,
    "low_signal_percentile": 12.0,
    "min_implant_fraction": 0.0005,
    "min_object_size": 24,
    "implant_dilation_px": 3,
}


def ensure_output_folders() -> None:
    """Create the expected output folders."""

    for folder in OUTPUT_FOLDERS.values():
        folder.mkdir(parents=True, exist_ok=True)


def protocol_with_dicom_metadata(protocol: dict, metadata: dict) -> dict:
    """Return a protocol copy using DICOM fields where they are available.

    DICOM bandwidth fields are vendor-dependent and often represent Hz per
    pixel rather than total receiver bandwidth, so the protocol template keeps
    its explicit total-bandwidth assumption.
    """

    merged = dict(protocol)
    field_map = {
        "field_strength_t": "field_strength_t",
        "tr_ms": "tr_ms",
        "te_ms": "te_ms",
        "flip_angle_deg": "flip_angle_deg",
    }
    for protocol_key, metadata_key in field_map.items():
        value = metadata.get(metadata_key)
        if value not in (None, ""):
            try:
                merged[protocol_key] = float(value)
            except (TypeError, ValueError):
                pass
    return merged
