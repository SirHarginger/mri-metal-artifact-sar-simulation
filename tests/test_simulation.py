from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LOW_SAR_PROTOCOL
from src.simulation import apply_simplified_artifact_correction


def test_inverse_simulation_correction_compensates_dark_signal_void():
    image = np.ones((48, 48), dtype=np.float32) * 0.65
    image[20:28, 20:28] = 0.02
    implant = np.zeros_like(image, dtype=bool)
    implant[20:28, 20:28] = True
    frequency = np.zeros_like(image, dtype=np.float32)

    corrected, info = apply_simplified_artifact_correction(
        image,
        implant,
        frequency,
        LOW_SAR_PROTOCOL,
    )

    assert corrected.shape == image.shape
    assert np.isfinite(corrected).all()
    assert corrected[implant].mean() > image[implant].mean()
    assert info["corrected_void_pixels"] >= implant.sum()
    assert info["frequency_bins"] == 5
