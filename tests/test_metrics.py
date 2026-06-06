from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import build_metrics_summary, compute_cnr, compute_snr


def test_metric_functions_return_finite_values():
    image = np.ones((24, 24), dtype=np.float32) * 0.6
    artifact = image.copy()
    artifact[10:14, 10:14] = 0.05
    corrected = artifact.copy()
    corrected[9:15, 9:15] = 0.4

    implant = np.zeros_like(image, dtype=bool)
    implant[10:14, 10:14] = True
    body = np.zeros_like(image, dtype=bool)
    body[3:21, 3:21] = True
    tissue = body & ~implant
    background = ~body
    near = np.zeros_like(image, dtype=bool)
    near[8:16, 8:16] = True
    near &= tissue

    masks = {
        "body": body,
        "implant": implant,
        "tissue": tissue,
        "background": background,
    }
    sar_standard = np.where(tissue, 1.0, 0.0).astype(np.float32)
    sar_low = np.where(tissue, 0.3, 0.0).astype(np.float32)

    snr = compute_snr(image, tissue, background)
    cnr = compute_cnr(image, tissue, implant, background)
    table = build_metrics_summary(
        image,
        artifact,
        artifact * 0.95,
        artifact * 0.8,
        corrected,
        masks,
        near,
        sar_standard,
        sar_low,
        sar_low * 0.8,
        [1.0, 1.0],
    )

    assert np.isfinite(snr)
    assert np.isfinite(cnr)
    assert len(table) == 5
    assert np.isfinite(table["SNR"]).all()
    assert np.isfinite(table["Relative SAR"]).all()
    assert "SAR reduction vs standard (%)" in table.columns
    assert "Image quality preservation score" in table.columns
