# SAR Model And Assumptions

The SAR output in this repository is a relative SAR-like map for comparing MRI
protocol trends. It is not a clinical SAR certification model.

## Base Convention

The model uses the RMS electric field convention:

```text
SAR = sigma * |E_rms|^2 / rho
```

where:

- `sigma` is simplified tissue conductivity in S/m.
- `E_rms` is a model-based RMS electric field magnitude.
- `rho` is tissue density in kg/m3.

The default values are defined in `src/config.py`.

## Protocol Scaling

The base map is scaled by protocol parameters:

```text
SAR_relative =
(sigma * |E_rms|^2 / rho)
* (alpha / alpha_ref)^2
* (TR_ref / TR)
* (beta / beta_ref)^2
* sequence_factor
* (B0 / B0_ref)^2
```

where:

- `alpha` is excitation flip angle.
- `beta` is refocusing angle.
- `TR` is repetition time.
- `sequence_factor` represents relative RF burden.
- `B0` is scanner field strength.

## Standard Protocol Template

```text
Field strength: 1.5 T
TR: 2500 ms
TE: 8-15 ms
Flip angle: 90 degrees
Refocusing angle: 180 degrees
Bandwidth: 200 kHz
Sequence: FSE/TSE
```

## Low-SAR Protocol Template

```text
Field strength: 1.5 T
Flip angle: reduced
Refocusing angle: 110-130 degrees
TR: increased
Bandwidth: increased
Parallel imaging: enabled
Fat suppression: STIR preferred
Sequence: FSE/TSE; SEMAC/MAVRIC if available or simulated
```

## Limitations

This repository does not claim to clinically validate true local SAR around
titanium implants. It does not model patient-specific coil loading, full
electromagnetic fields, implant geometry in 3D, thermoregulation, or scanner
safety limits. The output is intended for comparative protocol optimisation and
image quality analysis only.
