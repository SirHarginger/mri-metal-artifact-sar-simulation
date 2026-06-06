"""Run the MRI DICOM metal artefact and relative SAR pipeline."""

from __future__ import annotations

import pandas as pd

from src.config import (
    DICOM_PATH,
    FIGURES_DIR,
    LOW_SAR_PROTOCOL,
    MAPS_DIR,
    MASK_DIR,
    METRICS_DIR,
    REPORTS_DIR,
    STANDARD_PROTOCOL,
    ensure_output_folders,
    protocol_with_dicom_metadata,
)
from src.io import load_dicom, save_json, write_pipeline_report
from src.metrics import build_metrics_summary
from src.optimization import (
    optimization_summary_table,
    protocol_comparison_table,
    run_parameter_sweep,
    select_optimized_protocol,
)
from src.preprocessing import normalize_intensity
from src.sar_model import (
    compute_relative_sar_map,
    estimate_e_rms_map,
    summarize_sar_maps,
)
from src.segmentation import load_manual_mask, near_implant_ring, segment_masks
from src.simulation import (
    apply_simplified_artifact_correction,
    create_susceptibility_map,
    frequency_shift_hz,
    simulate_delta_b0_map,
    simulate_metal_artifact,
)
from src.visualization import (
    save_contour_overlay,
    save_comparison_panel,
    save_image,
    save_map,
    save_overlay,
)


def _metrics_markdown_table(metrics_df: pd.DataFrame) -> str:
    columns = [
        "Protocol",
        "Relative SAR",
        "SAR reduction vs standard (%)",
        "SNR",
        "CNR",
        "Artefact area",
        "Artefact reduction vs optimized (%)",
        "Image quality preservation score",
        "Comment",
    ]
    rows = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in metrics_df[columns].iterrows():
        rows.append(
            "| "
            + " | ".join(
                [
                    str(row["Protocol"]),
                    f"{float(row['Relative SAR']):.6g}",
                    f"{float(row['SAR reduction vs standard (%)']):.2f}",
                    f"{float(row['SNR']):.3f}",
                    f"{float(row['CNR']):.3f}",
                    f"{float(row['Artefact area']):.3f}",
                    f"{float(row['Artefact reduction vs optimized (%)']):.2f}",
                    f"{float(row['Image quality preservation score']):.3f}",
                    str(row["Comment"]),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def run_pipeline() -> dict:
    """Execute the full DICOM-to-output pipeline."""

    ensure_output_folders()

    image, metadata, _dataset = load_dicom(DICOM_PATH)
    save_json(metadata, REPORTS_DIR / "dicom_metadata.json")

    normalized = normalize_intensity(image)
    save_image(image, FIGURES_DIR / "original_dicom.png", "Original DICOM", cmap="gray")
    save_image(
        normalized,
        FIGURES_DIR / "normalized_image.png",
        "Normalized image",
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    manual_mask = load_manual_mask(MASK_DIR, DICOM_PATH, normalized.shape)
    masks, segmentation_info = segment_masks(normalized, manual_mask)
    ring_mask = near_implant_ring(masks["implant"], masks["tissue"], radius_px=8)

    save_image(masks["implant"], FIGURES_DIR / "implant_mask.png", "Implant/signal-void mask")
    save_image(masks["tissue"], FIGURES_DIR / "tissue_mask.png", "Tissue mask")
    save_image(
        masks["background"],
        FIGURES_DIR / "background_noise_mask.png",
        "Background/noise mask",
    )
    save_overlay(
        normalized,
        masks["implant"],
        FIGURES_DIR / "implant_mask_overlay.png",
        "Implant/signal-void mask overlay",
    )

    standard_protocol = protocol_with_dicom_metadata(STANDARD_PROTOCOL, metadata)
    low_sar_protocol = dict(LOW_SAR_PROTOCOL)
    low_sar_protocol["field_strength_t"] = standard_protocol["field_strength_t"]

    susceptibility_map = create_susceptibility_map(masks["implant"], masks["tissue"])
    delta_b0_map = simulate_delta_b0_map(
        susceptibility_map,
        masks["implant"],
        field_strength_t=standard_protocol["field_strength_t"],
    )
    frequency_map = frequency_shift_hz(delta_b0_map)

    save_map(
        susceptibility_map,
        MAPS_DIR / "susceptibility_map.png",
        "Susceptibility map",
        cmap="magma",
        colorbar_label="ppm",
    )
    save_map(
        delta_b0_map,
        MAPS_DIR / "delta_b0_map.png",
        "Delta-B0 map",
        cmap="coolwarm",
        colorbar_label="T",
    )
    save_map(
        frequency_map,
        MAPS_DIR / "frequency_shift_map.png",
        "Frequency shift map",
        cmap="coolwarm",
        colorbar_label="Hz",
    )

    e_rms_map = estimate_e_rms_map(normalized.shape, masks["implant"], masks["tissue"])
    standard_sar_map = compute_relative_sar_map(
        e_rms_map,
        masks["tissue"],
        standard_protocol,
        reference_protocol=standard_protocol,
    )
    low_sar_map = compute_relative_sar_map(
        e_rms_map,
        masks["tissue"],
        low_sar_protocol,
        reference_protocol=standard_protocol,
    )
    sar_summary = summarize_sar_maps(standard_sar_map, low_sar_map, masks["tissue"])

    sweep_df = run_parameter_sweep(
        e_rms_map,
        masks["tissue"],
        reference_protocol=standard_protocol,
    )
    optimized_protocol, optimized_summary = select_optimized_protocol(
        sweep_df,
        standard_protocol,
        sar_summary["standard_mean_relative_sar"],
    )
    optimized_sar_map = compute_relative_sar_map(
        e_rms_map,
        masks["tissue"],
        optimized_protocol,
        reference_protocol=standard_protocol,
    )
    optimized_values = optimized_sar_map[masks["tissue"]]
    optimized_summary["mean_relative_sar"] = (
        float(optimized_values.mean()) if optimized_values.size else 0.0
    )
    optimized_summary["max_relative_sar"] = (
        float(optimized_values.max()) if optimized_values.size else 0.0
    )
    if sar_summary["standard_mean_relative_sar"] > 0:
        optimized_summary["sar_reduction_percent"] = (
            100.0
            * (
                sar_summary["standard_mean_relative_sar"]
                - optimized_summary["mean_relative_sar"]
            )
            / sar_summary["standard_mean_relative_sar"]
        )

    standard_artifact, _standard_artifact_info = simulate_metal_artifact(
        normalized,
        masks["implant"],
        frequency_map,
        standard_protocol,
    )
    low_sar_artifact, _low_artifact_info = simulate_metal_artifact(
        normalized,
        masks["implant"],
        frequency_map,
        low_sar_protocol,
    )
    optimized_artifact, _optimized_artifact_info = simulate_metal_artifact(
        normalized,
        masks["implant"],
        frequency_map,
        optimized_protocol,
    )
    corrected, correction_info = apply_simplified_artifact_correction(
        optimized_artifact,
        masks["implant"],
        frequency_map,
        optimized_protocol,
    )

    save_image(
        standard_artifact,
        FIGURES_DIR / "artifact_image.png",
        "Artefact-corrupted image",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    save_image(
        low_sar_artifact,
        FIGURES_DIR / "low_sar_artifact_image.png",
        "Low-SAR artefact simulation",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    save_image(
        optimized_artifact,
        FIGURES_DIR / "optimized_low_sar_artifact_image.png",
        "Optimized low-SAR artefact simulation",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    save_image(
        corrected,
        FIGURES_DIR / "corrected_image.png",
        "Metal artifact-corrected image",
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    artifact_reduction_map = abs(optimized_artifact - corrected)
    save_map(
        artifact_reduction_map,
        MAPS_DIR / "artifact_reduction_map.png",
        "Artifact reduction map",
        cmap="viridis",
        colorbar_label="absolute intensity correction",
    )
    save_contour_overlay(
        corrected,
        masks["implant"],
        FIGURES_DIR / "artifact_correction_overlay.png",
        "Artifact correction region overlay",
    )
    save_contour_overlay(
        corrected,
        masks["implant"],
        FIGURES_DIR / "implant_clear_image.png",
        "Artifact correction region overlay",
    )
    save_comparison_panel(
        [normalized, low_sar_artifact, optimized_artifact, corrected],
        ["Original DICOM", "Low-SAR template", "Optimized low-SAR", "Artifact correction"],
        FIGURES_DIR / "comparison_panel.png",
    )

    save_map(
        standard_sar_map,
        MAPS_DIR / "standard_relative_sar_map.png",
        "Standard relative SAR-like map",
        cmap="inferno",
        colorbar_label="relative SAR-like value",
    )
    save_map(
        low_sar_map,
        MAPS_DIR / "low_sar_relative_sar_map.png",
        "Low-SAR relative SAR-like map",
        cmap="inferno",
        colorbar_label="relative SAR-like value",
    )
    save_map(
        optimized_sar_map,
        MAPS_DIR / "optimized_relative_sar_map.png",
        "Optimized relative SAR-like map",
        cmap="inferno",
        colorbar_label="relative SAR-like value",
    )

    metrics_df = build_metrics_summary(
        normalized,
        standard_artifact,
        low_sar_artifact,
        optimized_artifact,
        corrected,
        masks,
        ring_mask,
        standard_sar_map,
        low_sar_map,
        optimized_sar_map,
        metadata.get("pixel_spacing_mm"),
    )
    metrics_df.to_csv(METRICS_DIR / "metrics_summary.csv", index=False)

    comparison_df = protocol_comparison_table(
        sar_summary,
        standard_protocol,
        low_sar_protocol,
    )
    comparison_df.to_csv(METRICS_DIR / "protocol_comparison.csv", index=False)

    sweep_df.to_csv(METRICS_DIR / "parameter_sweep_results.csv", index=False)

    optimization_df = optimization_summary_table(
        standard_protocol,
        low_sar_protocol,
        optimized_protocol,
        sar_summary["standard_mean_relative_sar"],
        sar_summary["low_sar_mean_relative_sar"],
        optimized_summary,
    )
    optimization_df.to_csv(METRICS_DIR / "optimization_summary.csv", index=False)

    report_metadata = dict(metadata)
    report_metadata["correction_method"] = correction_info["method"]
    report_metadata["optimized_protocol"] = optimized_summary
    write_pipeline_report(
        REPORTS_DIR / "pipeline_report.md",
        report_metadata,
        segmentation_info,
        sar_summary,
        _metrics_markdown_table(metrics_df),
    )

    return {
        "metadata": metadata,
        "segmentation": segmentation_info,
        "sar_summary": sar_summary,
        "metrics": metrics_df,
        "outputs": {
            "figures": str(FIGURES_DIR),
            "maps": str(MAPS_DIR),
            "metrics": str(METRICS_DIR),
            "reports": str(REPORTS_DIR),
        },
    }


if __name__ == "__main__":
    result = run_pipeline()
    print("Pipeline complete.")
    print(f"Figures: {result['outputs']['figures']}")
    print(f"Maps: {result['outputs']['maps']}")
    print(f"Metrics: {result['outputs']['metrics']}")
    print(f"Reports: {result['outputs']['reports']}")
