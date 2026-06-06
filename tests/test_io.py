from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DICOM_PATH
from src.io import load_dicom


def test_load_dicom_returns_2d_image_and_metadata():
    assert Path(DICOM_PATH).exists()

    image, metadata, _dataset = load_dicom(DICOM_PATH)

    assert image.ndim == 2
    assert image.size > 0
    assert np.isfinite(image).all()
    assert isinstance(metadata, dict)
    assert metadata["image_size"] == [image.shape[0], image.shape[1]]
