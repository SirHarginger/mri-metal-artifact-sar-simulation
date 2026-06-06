"""Plotting helpers for automatic pipeline outputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mri_metal_artifact_matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi


def _prepare_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_image(
    image: np.ndarray,
    output_path: str | Path,
    title: str,
    cmap: str = "gray",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Save a single image without axes."""

    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_map(
    data: np.ndarray,
    output_path: str | Path,
    title: str,
    cmap: str = "viridis",
    colorbar_label: str | None = None,
) -> None:
    """Save a technical map with a colorbar."""

    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    image = ax.imshow(data, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if colorbar_label:
        colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Save an image with a red mask overlay."""

    path = _prepare_path(output_path)
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask.astype(np.float32) * 0.45
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_contour_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    """Save a corrected image with an artefact-correction region contour."""

    path = _prepare_path(output_path)
    edge = ndi.binary_dilation(mask.astype(bool), iterations=1) ^ ndi.binary_erosion(
        mask.astype(bool),
        iterations=1,
    )
    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.imshow(image, cmap="gray", vmin=0, vmax=1)
    overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    overlay[..., 0] = 1.0
    overlay[..., 1] = 0.85
    overlay[..., 3] = edge.astype(np.float32) * 0.95
    ax.imshow(overlay)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_comparison_panel(
    images: list[np.ndarray],
    titles: list[str],
    output_path: str | Path,
    cmap: str = "gray",
) -> None:
    """Save a side-by-side comparison panel."""

    path = _prepare_path(output_path)
    count = len(images)
    fig, axes = plt.subplots(1, count, figsize=(4 * count, 4), dpi=160)
    if count == 1:
        axes = [axes]
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
