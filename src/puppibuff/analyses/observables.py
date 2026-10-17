from __future__ import annotations

from ..utils import Source, bool_mask

import numpy as np
import vector

from numpy.typing import NDArray
from typing import Any

#-----------------------------------------------------------------------------

def _four_momenta(source: Source) -> Any:
    """`source`'s consistuents as massless four-momenta"""
                                        # float64: rapidity cancels at large |eta|
    channels = { channel: source[channel].astype(np.float64) for channel in ( "pt", "eta", "phi" ) }

    return vector.array(channels | { "mass": np.zeros_like(channels["pt"]) })


def pairwise_delta_r(source: Source) -> NDArray:
    """Compute Delta R between every pair of genuine constituents in the same
    jet, in rapidity (see `axis_delta_r`).
    """
    constituents = _four_momenta(source)
    real = bool_mask(source["real"])
                                        # Every pair of slots once
    first, second = np.triu_indices(real.shape[1], k = 1)

    delta_r = constituents[:, first].deltaRapidityPhi(constituents[:, second])

    return delta_r[real[:, first] & real[:, second]]


def axis_delta_r(source: Source) -> NDArray:
    """Compute Delta R between each genuine constituent and its jet axis as the 
    summed four-momentum of the jet's constituents. Using rapidity instead of 
    pseudo- as FastJet clustered the jets using rapidity.
    """
    constituents = _four_momenta(source)
    axis = constituents.sum(axis = 1)   # (N,)

    return constituents.deltaRapidityPhi(axis[:, None])[bool_mask(source["real"])]
