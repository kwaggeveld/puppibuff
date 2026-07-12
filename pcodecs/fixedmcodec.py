from .codec import Codec
from ..datasets import Dataset

import numpy as np

from numpy.typing import NDArray


class FixedMCodec(Codec):
    """Per-channel normaliser for fixed multiplicity M PuppiJet events.
    
    pt  -> log1p -> normalise (over all jets)
    eta -> normalise          (over all jets)
    phi -> (sin_phi, cos_phi)

    Output: array containing N encoded events, shape (4, N).
        [pt, eta, sin_phi, cos_phi]
    """
    
    s_REQUIRED_CHANNELS = ["pt", "eta", "phi"]
    s_KEYS = ["pt_mean", "pt_std", "eta_mean", "eta_std"]

    def fit(self, data: Dataset) -> None:
        self.check_dataset(data)

        logpt = np.log1p(data["pt"])
        eta = data["eta"]

        self.pt_mean  = float(logpt.mean())
        self.pt_std   = float(logpt.std())
        self.eta_mean = float(eta.mean())
        self.eta_std  = float(eta.std())

    def encode(self, data: Dataset) -> NDArray:
        self.check_dataset(data)

        std_pt  = (np.log1p(data["pt"]) - self.pt_mean) / self.pt_std
        std_eta = (data["eta"] - self.eta_mean) / self.eta_std
        sin_phi = np.sin(data["phi"])
        cos_phi = np.cos(data["phi"])

        # from 4 x (# events, M) to (# events, 4, M)
        return np.stack([std_pt, std_eta, sin_phi, cos_phi], axis = 1)

    def decode(self, out: NDArray) -> dict[str, NDArray]:
        return {
            "pt":  np.expm1(out["pt"] * self.pt_std + self.pt_mean),
            "eta": out["eta"] * self.eta_std + self.eta_mean,
            "phi": np.arctan2(out["sin_phi"], out["cos_phi"]),  # -> (-pi, pi]
        }