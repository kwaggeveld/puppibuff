from __future__ import annotations

from .npydirdataset import NpyDirDataset
from ..utils import bool_mask

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class L1Puppi(NpyDirDataset):
    """Pre-clustered, zero-padded L1Puppi jet constituents relative to their
    jet: pt / jet_pt, eta - jet_eta and phi - jet_phi wrapped to [-pi, pi).

    Clustering + padding are done by `scripts/cluster_l1puppi.py`: each .npy file
    holds dense (n_jets, m) arrays per channel (jets padded/truncated to a fixed
    m), plus a `real` mask (1 = genuine constituent, 0 = padding), and the
    jets' own (n_jets,) kinematics over all its constituents, kept as (N,)
    channels.
    """

    s_JET_KEYS = [ "jet_pt", "jet_eta", "jet_phi" ]
    s_CHANNELS = [ "pt", "eta", "phi", "real" ] + s_JET_KEYS
    s_LOCATION_ENV = "CLUSTERED_L1PUPPI_LOCATION"

    s_LOG_Y = { "eta", "phi" }          # Peaked at the axis, long tails

    def _select(self, data: dict) -> dict[str, NDArray]:
        """One file's constituents, relative to the file's jet kinematics."""
        if missing := set(self.s_JET_KEYS) - set(data):
            raise KeyError(f"No { sorted(missing) } in { self.dir }. Cluster "
                           f"with `scripts/cluster_l1puppi.py`.")

                                        # (N, 1), to broadcast over the slots
        jet_pt, jet_eta, jet_phi = ( data[key][:, None] for key in self.s_JET_KEYS )

        relative = {
            "pt":  data["pt"] / jet_pt,
            "eta": data["eta"] - jet_eta,
            "phi": (data["phi"] - jet_phi + np.pi) % (2 * np.pi) - np.pi,
        }
        
        real = bool_mask(data["real"])

        return { channel: np.where(real, values, 0) for channel, values in relative.items() } \
               | { "real": data["real"] } \
               | { key: data[key] for key in self.s_JET_KEYS }
