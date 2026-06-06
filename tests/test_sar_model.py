from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import LOW_SAR_PROTOCOL, STANDARD_PROTOCOL
from src.sar_model import compute_relative_sar_map, estimate_e_rms_map


def test_low_sar_protocol_reduces_relative_sar():
    shape = (32, 32)
    implant = np.zeros(shape, dtype=bool)
    implant[14:18, 14:18] = True
    tissue = ~implant

    e_rms = estimate_e_rms_map(shape, implant, tissue)
    standard = compute_relative_sar_map(
        e_rms,
        tissue,
        STANDARD_PROTOCOL,
        reference_protocol=STANDARD_PROTOCOL,
    )
    low = compute_relative_sar_map(
        e_rms,
        tissue,
        LOW_SAR_PROTOCOL,
        reference_protocol=STANDARD_PROTOCOL,
    )

    assert standard.shape == shape
    assert low.shape == shape
    assert np.nanmean(low[tissue]) < np.nanmean(standard[tissue])
    assert np.isfinite(standard).all()
    assert np.isfinite(low).all()
