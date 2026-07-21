from __future__ import annotations

import numpy as np

from numpy.typing import NDArray



def pt_power_weights(
    pt: NDArray,
    alpha: float = 0.3,
    cap_percentile: float = 99.9,
) -> NDArray:
    """Per-event weight ~ pt**alpha, saturating above `cap_percentile`
    to prevent extreme events from getting extreme weight"""
    pt_cap = np.percentile(pt, cap_percentile)

                                        # Median pt has weight 1
    weight = (np.minimum(pt, pt_cap) / np.median(pt)) ** alpha
    return (weight / weight.mean()).astype(np.float32)