"""Simplified susceptibility, field perturbation, artefact, and correction models."""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import fftconvolve
from skimage import morphology

from src.config import GYROMAGNETIC_RATIO_HZ_PER_T, SUSCEPTIBILITY


def create_susceptibility_map(
    implant_mask: np.ndarray,
    tissue_mask: np.ndarray,
    tissue_ppm: float = SUSCEPTIBILITY["tissue_ppm"],
    implant_ppm: float = SUSCEPTIBILITY["titanium_implant_ppm"],
) -> np.ndarray:
    """Assign simplified relative magnetic susceptibility values in ppm."""

    susceptibility = np.zeros_like(implant_mask, dtype=np.float32)
    susceptibility[tissue_mask] = tissue_ppm
    susceptibility[implant_mask] = implant_ppm
    return susceptibility


def _dipole_like_kernel(size: int) -> np.ndarray:
    size = int(size)
    if size % 2 == 0:
        size += 1
    coords = np.arange(size) - size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    radius2 = xx**2 + yy**2
    radius2[size // 2, size // 2] = 1.0
    kernel = (3.0 * yy**2 / radius2 - 1.0) / np.power(radius2, 1.5)
    kernel[size // 2, size // 2] = 0.0
    kernel -= kernel.mean()
    max_abs = np.max(np.abs(kernel))
    if max_abs > 0:
        kernel /= max_abs
    return kernel.astype(np.float32)


def simulate_delta_b0_map(
    susceptibility_ppm: np.ndarray,
    implant_mask: np.ndarray,
    field_strength_t: float,
) -> np.ndarray:
    """Generate a simplified implant-induced delta-B0 map in tesla.

    This is a compact 2D approximation: an implant mask is convolved with a
    dipole-like kernel and scaled by the susceptibility contrast and B0. It is
    useful for comparative artefact behavior, not for quantitative field maps.
    """

    delta_chi = float(np.max(susceptibility_ppm) - np.min(susceptibility_ppm)) * 1e-6
    kernel_size = max(31, min(129, int(min(susceptibility_ppm.shape) * 0.25)))
    kernel = _dipole_like_kernel(kernel_size)
    field_pattern = fftconvolve(implant_mask.astype(np.float32), kernel, mode="same")
    max_abs = np.max(np.abs(field_pattern))
    if max_abs > 0:
        field_pattern /= max_abs

    delta_b0 = float(field_strength_t) * delta_chi * field_pattern
    delta_b0 = ndi.gaussian_filter(delta_b0, sigma=1.0)
    delta_b0[implant_mask] = float(field_strength_t) * delta_chi
    return delta_b0.astype(np.float32)


def frequency_shift_hz(delta_b0_t: np.ndarray) -> np.ndarray:
    """Convert delta-B0 in tesla to off-resonance frequency shift in hertz."""

    return (np.asarray(delta_b0_t, dtype=np.float32) * GYROMAGNETIC_RATIO_HZ_PER_T).astype(
        np.float32
    )


def _artifact_severity(protocol: dict) -> float:
    te_scale = float(protocol.get("te_ms", 12.0)) / 12.0
    b0_scale = float(protocol.get("field_strength_t", 1.5)) / 1.5
    bandwidth_scale = 200.0 / max(float(protocol.get("bandwidth_khz", 200.0)), 1.0)
    return max(0.15, te_scale * b0_scale * bandwidth_scale)


def _warp_by_frequency(
    image: np.ndarray,
    frequency_shift: np.ndarray,
    protocol: dict,
    direction: float = 1.0,
) -> np.ndarray:
    rows, cols = image.shape
    yy, xx = np.indices(image.shape, dtype=np.float32)
    bandwidth_hz = max(float(protocol.get("bandwidth_khz", 200.0)) * 1000.0, 1.0)
    shift_px = direction * (frequency_shift / bandwidth_hz) * cols * 0.08
    shift_px = np.clip(shift_px, -14.0, 14.0)
    return ndi.map_coordinates(
        image,
        [yy, xx - shift_px],
        order=1,
        mode="nearest",
    ).astype(np.float32)


def simulate_metal_artifact(
    normalized_image: np.ndarray,
    implant_mask: np.ndarray,
    frequency_shift: np.ndarray,
    protocol: dict,
) -> tuple[np.ndarray, dict]:
    """Simulate simplified off-resonance metal artefacts.

    The model combines geometric distortion, signal void, local dephasing loss,
    blurring, and mild pile-up. Severity depends on TE, bandwidth, B0, and the
    implant-induced off-resonance map.
    """

    image = np.asarray(normalized_image, dtype=np.float32)
    severity = _artifact_severity(protocol)
    warped = _warp_by_frequency(image, frequency_shift, protocol, direction=1.0)

    abs_freq = np.abs(frequency_shift)
    freq_scale = np.percentile(abs_freq[abs_freq > 0], 95) if np.any(abs_freq > 0) else 1.0
    freq_norm = np.clip(abs_freq / max(freq_scale, 1.0), 0.0, 1.0)

    influence = morphology.dilation(
        implant_mask,
        morphology.disk(max(3, int(5 * severity))),
    )
    void_mask = morphology.dilation(
        implant_mask,
        morphology.disk(max(1, int(2 * severity))),
    )

    te_s = float(protocol.get("te_ms", 12.0)) / 1000.0
    dephasing_loss = np.exp(-freq_norm * severity * (1.0 + 40.0 * te_s))
    corrupted = warped * dephasing_loss

    blurred = ndi.gaussian_filter(corrupted, sigma=max(0.8, 1.8 * severity))
    corrupted = np.where(influence, 0.55 * corrupted + 0.45 * blurred, corrupted)
    corrupted[void_mask] *= 0.04

    pileup = ndi.gaussian_filter(image * influence.astype(np.float32), sigma=1.2)
    rim = morphology.dilation(void_mask, morphology.disk(3)) & ~void_mask
    corrupted[rim] += 0.12 * severity * pileup[rim]

    corrupted = np.clip(corrupted, 0.0, 1.0).astype(np.float32)
    components = {
        "severity": float(severity),
        "influence_mask": influence.astype(bool),
        "void_mask": void_mask.astype(bool),
        "frequency_norm": freq_norm.astype(np.float32),
    }
    return corrupted, components


def apply_simplified_artifact_correction(
    artifact_image: np.ndarray,
    implant_mask: np.ndarray,
    frequency_shift: np.ndarray,
    protocol: dict,
) -> tuple[np.ndarray, dict]:
    """Apply a simplified inverse-simulation metal artefact correction.

    This is not a vendor reconstruction and it does not recover true missing
    anatomy. It uses only the artefact-corrupted image plus the estimated
    off-resonance map to reverse the simulated distortion and attenuation:
    inverse geometric warping, MAVRIC-like frequency-bin recombination,
    high-bandwidth-like stabilization, and capped signal-loss compensation.
    """

    image = np.asarray(artifact_image, dtype=np.float32)
    severity = _artifact_severity(protocol)
    abs_freq = np.abs(frequency_shift)
    freq_scale = np.percentile(abs_freq[abs_freq > 0], 95) if np.any(abs_freq > 0) else 1.0
    freq_norm = np.clip(abs_freq / max(freq_scale, 1.0), 0.0, 1.0)

    direct_inverse = _warp_by_frequency(image, frequency_shift, protocol, direction=-1.0)

    bin_centers = np.linspace(-0.8, 0.8, 5, dtype=np.float32) * float(freq_scale)
    bin_width = max(float(freq_scale) * 0.45, 1.0)
    weighted_sum = np.zeros_like(image, dtype=np.float32)
    weight_sum = np.zeros_like(image, dtype=np.float32)
    for center in bin_centers:
        residual_frequency = frequency_shift - center
        candidate = _warp_by_frequency(
            image,
            residual_frequency,
            protocol,
            direction=-0.9,
        )
        weight = np.exp(-0.5 * np.square(residual_frequency / bin_width)).astype(np.float32)
        weighted_sum += candidate * weight
        weight_sum += weight
    mavric_like = weighted_sum / np.maximum(weight_sum, 1e-6)

    high_bandwidth_protocol = dict(protocol)
    high_bandwidth_protocol["bandwidth_khz"] = float(protocol.get("bandwidth_khz", 200.0)) * 2.0
    high_bandwidth_like = _warp_by_frequency(
        image,
        frequency_shift,
        high_bandwidth_protocol,
        direction=-0.55,
    )

    corrected = 0.50 * mavric_like + 0.35 * direct_inverse + 0.15 * high_bandwidth_like

    te_s = float(protocol.get("te_ms", 12.0)) / 1000.0
    dephasing_loss = np.exp(-freq_norm * severity * (1.0 + 40.0 * te_s))
    void_mask = morphology.dilation(
        implant_mask,
        morphology.disk(max(1, int(2 * severity))),
    )
    simulated_attenuation = dephasing_loss * np.where(void_mask, 0.04, 1.0)
    gain = 1.0 / np.clip(simulated_attenuation, 0.055, 1.0)
    gain = np.clip(gain, 1.0, 18.0).astype(np.float32)

    near_metal = morphology.dilation(
        implant_mask,
        morphology.disk(max(6, int(9 * severity))),
    )
    corrected = np.where(near_metal, corrected * gain, corrected)

    local_median = ndi.median_filter(corrected, size=5)
    local_smooth = ndi.gaussian_filter(corrected, sigma=1.0)
    sharpened = corrected + 0.35 * (corrected - local_smooth)
    corrected = np.where(near_metal, 0.70 * sharpened + 0.30 * local_median, corrected)

    cap_region = morphology.dilation(implant_mask, morphology.disk(24)) & ~implant_mask
    if cap_region.any():
        cap_values = image[cap_region]
    else:
        cap_values = image[np.isfinite(image)]
    if cap_values.size:
        low_cap, high_cap = np.percentile(cap_values, [2.0, 98.0])
        corrected[near_metal] = np.clip(
            corrected[near_metal],
            max(0.0, float(low_cap) * 0.5),
            min(1.0, float(high_cap) * 1.35),
        )

    corrected = np.clip(corrected, 0.0, 1.0).astype(np.float32)

    info = {
        "method": "inverse-simulation correction with MAVRIC-like frequency recombination and capped signal-loss compensation",
        "near_metal_pixels": int(near_metal.sum()),
        "corrected_void_pixels": int(void_mask.sum()),
        "frequency_bins": int(len(bin_centers)),
        "max_compensation_gain": float(np.max(gain[near_metal])) if near_metal.any() else 1.0,
    }
    return corrected, info
