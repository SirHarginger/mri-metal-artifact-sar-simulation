# MRI Metal Artefact And Relative SAR Simulation

This repository implements a Python-based simulation pipeline for studying
titanium/prosthetic implant-induced MRI artefacts and relative SAR reduction
under low-SAR protocol conditions. It uses a real MRI DICOM image as the first
dataset and is structured so additional DICOM files can be added later.

Important scientific scope: this project supports comparative protocol
optimisation using simplified model-based SAR trends and image quality metrics.
It does not clinically validate true local SAR around titanium implants and does
not replace scanner safety certification.

## Repository Layout

```text
data/dicom/      Real DICOM input files
data/masks/      Optional manual implant/signal-void masks
docs/            Technical documentation
outputs/figures/ Image previews, masks, overlays, comparisons
outputs/maps/    Susceptibility, field, frequency, and relative SAR maps
outputs/metrics/ CSV metrics and protocol sweep results
outputs/reports/ Metadata JSON and Markdown pipeline report
src/             Modular pipeline implementation
tests/           Basic pytest coverage
notebooks/       Minimal demo notebooks
```

The default dataset is:

```text
data/dicom/MR.0962_25.Image 2.0004.dcm
```

## Install

Create or activate a Python environment, then install the scientific imaging
dependencies:

```bash
pip install -r requirements.txt
```

The project uses:

```text
pydicom
numpy
scipy
scikit-image
matplotlib
pandas
pytest
jupyter
```

## Run The Pipeline

From the repository root:

```bash
python main.py
```

The run will:

1. Load the DICOM image with `pydicom`.
2. Extract image metadata and save it to `outputs/reports/dicom_metadata.json`.
3. Normalize image intensity to a robust 0-1 range.
4. Segment the implant/signal-void region using low-signal thresholding, or use
   an optional manual mask from `data/masks/`.
5. Generate tissue and background/noise masks.
6. Create a simplified susceptibility map, delta-B0 map, and frequency-shift map.
7. Simulate implant-related MRI artefacts from off-resonance effects.
8. Estimate comparative relative SAR-like maps for standard and low-SAR protocols.
9. Run a low-SAR parameter sweep.
10. Select an optimized low-SAR protocol from the sweep using SAR reduction,
   SNR-preservation, and artifact-risk proxies.
11. Correct the optimized low-SAR artefact image using inverse geometric
   warping, MAVRIC-like frequency-bin recombination, high-bandwidth-like
   stabilization, and capped signal-loss compensation.
11. Save figures, maps, metrics, and a Markdown report.

## Optional Manual Masks

Automatic segmentation is the default. If it performs poorly, place a binary
mask in `data/masks/` with a filename matching the DICOM stem or containing
`mask`. Supported formats are `.npy`, `.png`, `.jpg`, `.jpeg`, `.tif`, and
`.tiff`. Non-zero pixels are treated as implant/signal-void mask pixels.

## Outputs

Main output files include:

```text
outputs/figures/original_dicom.png
outputs/figures/normalized_image.png
outputs/figures/implant_mask_overlay.png
outputs/figures/artifact_image.png
outputs/figures/optimized_low_sar_artifact_image.png
outputs/figures/corrected_image.png
outputs/figures/artifact_correction_overlay.png
outputs/figures/comparison_panel.png

outputs/maps/susceptibility_map.png
outputs/maps/delta_b0_map.png
outputs/maps/frequency_shift_map.png
outputs/maps/artifact_reduction_map.png
outputs/maps/standard_relative_sar_map.png
outputs/maps/low_sar_relative_sar_map.png
outputs/maps/optimized_relative_sar_map.png

outputs/metrics/metrics_summary.csv
outputs/metrics/protocol_comparison.csv
outputs/metrics/parameter_sweep_results.csv
outputs/metrics/optimization_summary.csv

outputs/reports/dicom_metadata.json
outputs/reports/pipeline_report.md
```

## Relative SAR Model

The relative SAR-like map uses the RMS electric field convention:

```text
SAR = sigma * |E_rms|^2 / rho
```

It then applies protocol scaling for flip angle, refocusing angle, TR, sequence
RF burden, and field strength:

```text
SAR_relative =
(sigma * |E_rms|^2 / rho)
* (alpha / alpha_ref)^2
* (TR_ref / TR)
* (beta / beta_ref)^2
* sequence_factor
* (B0 / B0_ref)^2
```

The artifact-corrected image is produced from the simulated artefact image and the
estimated off-resonance model. It is not created by substituting pixels from the
original DICOM. It reduces the simulated dark signal void/shadow and clarifies
the artifact-affected region, but it does not recover true MRI signal from metal-hidden
anatomy.

This is a comparative research/teaching model. It is not a regulatory SAR
estimator, not patient-specific local SAR dosimetry, and not a clinical safety
tool.

## Tests

Run:

```bash
pytest
```

The tests check DICOM loading, relative SAR scaling, and image quality metric
functions.
