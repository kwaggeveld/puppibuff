from .codec import Codec
from ..datasets import Dataset

import numpy as np

from numpy.typing import NDArray


class FixedMCodec(Codec):
    """Per-channel normaliser for fixed multiplicity M PuppiJet events.

    pt  -> log1p -> normalise (over all jets)
    eta -> normalise          (over all jets)
    phi -> (sin_phi, cos_phi)

    ``encode`` stacks the 4 channels on axis 1, so an input of N events with
    multiplicity M maps to shape (N, 4, M) (or (N, 4) for flattened jets).
    ``decode`` reads the same layout back, indexing the channels on axis 1.
    """

    s_REQUIRED_CHANNELS = [ "pt", "eta", "phi" ]
    s_KEYS = [ channel + "_" + attr
               for channel in ( "pt", "eta" )
               for attr    in ( "mean", "std", "min", "max" )]

    def fit(self, data: Dataset) -> None:
        self.check_dataset(data)

        pt  = data['pt']
        eta = data["eta"]
        
        logpt = np.log1p(pt)

        self.pt_mean  = float(logpt.mean())
        self.pt_std   = float(logpt.std())
        self.eta_mean = float(eta.mean())
        self.eta_std  = float(eta.std())

        # Observed physical ranges to clip samples to
        self.pt_min  = float(np.nanmin(pt))
        self.pt_max  = float(np.nanmax(pt))
        self.eta_min = float(np.nanmin(eta))
        self.eta_max = float(np.nanmax(eta))

    def encode(self, data: NDarray) -> NDArray:
        self.check_dataset(data)

        std_pt  = (np.log1p(data["pt"]) - self.pt_mean) / self.pt_std
        std_eta = (data["eta"] - self.eta_mean) / self.eta_std
        sin_phi = np.sin(data["phi"])
        cos_phi = np.cos(data["phi"])

        # from 4 x (N, M) to (N, 4, M)
        return np.stack([std_pt, std_eta, sin_phi, cos_phi], axis = 1)

    def decode(self, out: NDArray) -> dict[str, NDArray]:
        std_pt, std_eta, sin_phi, cos_phi = np.moveaxis(out, 1, 0)

        pt  = np.expm1(std_pt * self.pt_std + self.pt_mean)
        eta = std_eta * self.eta_std + self.eta_mean

        return {
            "pt":  np.clip(pt,  self.pt_min,  self.pt_max),     # to observed range
            "eta": np.clip(eta, self.eta_min, self.eta_max),
            "phi": np.arctan2(sin_phi, cos_phi),                # -> (-pi, pi]
        }