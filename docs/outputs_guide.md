# Outputs Guide

All outputs are written automatically when `python main.py` is run.

## Figures

Saved in `outputs/figures/`.

- `original_dicom.png`: preview of the raw DICOM pixel array.
- `normalized_image.png`: robust 0-1 normalized image.
- `implant_mask.png`: binary implant/signal-void mask.
- `tissue_mask.png`: tissue/body mask excluding implant/signal void.
- `background_noise_mask.png`: background/noise mask used for metrics.
- `implant_mask_overlay.png`: implant/signal-void mask overlaid on the image.
- `artifact_image.png`: simulated standard-protocol artefact image.
- `low_sar_artifact_image.png`: simulated low-SAR artefact image.
- `optimized_low_sar_artifact_image.png`: simulated artefact image for the
  selected optimized low-SAR protocol.
- `corrected_image.png`: inverse-simulation corrected image generated from the
  optimized simulated artefact image.
- `artifact_correction_overlay.png`: corrected image with a contour around the
  estimated artifact-correction region.
- `comparison_panel.png`: original, standard artefact, low-SAR artefact, and
  corrected images side by side.

## Maps

Saved in `outputs/maps/`.

- `susceptibility_map.png`: simplified tissue/implant susceptibility assignment.
- `delta_b0_map.png`: simplified implant-induced magnetic field perturbation.
- `frequency_shift_map.png`: delta-B0 converted to off-resonance frequency.
- `artifact_reduction_map.png`: absolute intensity difference between the
  low-SAR artefact image and the corrected image.
- `standard_relative_sar_map.png`: relative SAR-like map for the standard
  protocol.
- `low_sar_relative_sar_map.png`: relative SAR-like map for the low-SAR
  protocol.
- `optimized_relative_sar_map.png`: relative SAR-like map for the selected
  optimized low-SAR protocol.

## Metrics

Saved in `outputs/metrics/`.

- `metrics_summary.csv`: SNR, CNR, artefact area, artifact reduction percentage,
  sharpness, near-implant signal, signal loss, max relative SAR, mean relative
  SAR, and comments for standard, low-SAR, artefact-corrupted, and corrected
  outputs.
- `protocol_comparison.csv`: compact comparison of standard and low-SAR protocol
  settings and relative SAR reduction.
- `parameter_sweep_results.csv`: sweep over flip angle, refocusing angle, TR, TE,
  bandwidth, and sequence factor.
- `optimization_summary.csv`: standard, low-SAR template, and selected
  optimized low-SAR protocol settings with SAR/image-quality proxies.

## Reports

Saved in `outputs/reports/`.

- `dicom_metadata.json`: extracted DICOM metadata summary.
- `pipeline_report.md`: human-readable report with DICOM metadata, segmentation
  details, relative SAR summary, metrics table, and output locations.

The SAR reports are comparative model outputs only and are not clinical SAR
safety certification.
