# Pipeline Overview

This project runs a real DICOM-first MRI processing pipeline for studying
prosthetic/titanium implant-related artefacts and comparative low-SAR protocol
behaviour.

## Workflow

1. `main.py` loads `data/dicom/MR.0962_25.Image 2.0004.dcm` with `pydicom`.
2. Pixel data are converted to a 2D floating-point image and useful metadata are
   extracted, including image size, pixel spacing, sequence labels, TR, TE, flip
   angle, field strength, pixel bandwidth, and scanner-reported SAR when present.
3. The image is normalized to a robust 0-1 intensity range using percentile
   clipping.
4. The implant/signal-void region is segmented as a low-signal region inside the
   estimated body/tissue mask. Optional manual masks in `data/masks/` override
   automatic segmentation.
5. Tissue and background/noise masks are generated for image metrics.
6. A simplified susceptibility map assigns tissue and titanium/prosthetic mask
   values.
7. A compact dipole-like 2D convolution creates a simplified delta-B0 map, which
   is converted to an off-resonance frequency-shift map.
8. Artefacts are simulated using geometric distortion, signal void, intensity
   loss, local blur, and mild pile-up. TE, bandwidth, field strength, and the
   implant mask influence severity.
9. Relative SAR-like maps are calculated for standard and low-SAR protocol
   templates.
10. A parameter sweep explores flip angle, refocusing angle, TR, TE, bandwidth,
    and sequence factor combinations.
11. An optimized low-SAR protocol is selected using a balanced score for SAR
    reduction, SNR preservation, and artifact-risk control.
12. The simulated optimized low-SAR artefact image is corrected using an inverse
    simulation: estimated inverse geometric warping, MAVRIC-like frequency-bin
    recombination, high-bandwidth-like stabilization, and capped signal-loss
    compensation. A separate contour figure marks the artifact-correction
    region.
13. Figures, maps, CSV metrics, metadata JSON, and a Markdown report are saved
    under `outputs/`.

## Scientific Scope

The pipeline is designed for comparative image-processing and protocol-trend
analysis. The corrected image is generated from the simulated artefact image and
estimated maps, not by substituting the original DICOM into the mask. It does
not recover signal that was never acquired. The project does not perform full
electromagnetic simulation, full MRI sequence reconstruction, vendor
MAVRIC/SEMAC reconstruction, or clinical SAR validation.
