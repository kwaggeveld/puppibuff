from .codec import Codec
from ..datasets import Dataset

import numpy as np

from numpy.typing import NDArray


class FixedMCodec(Codec):
    """Per-channel normaliser for fixed multiplicity M PuppiJet events.

    pt  -> log1p -> normalise (over all jets)
    eta -> normalise          (over all jets)
    phi -> (sin_phi, cos_phi)
    """

    s_EXPORT_KEYS = [ channel + "_" + attr
                      for channel in ( "pt", "eta" )
                      for attr    in ( "mean", "std", "min", "max" )]
    
    def check_dataset(self, data: Dataset) -> None:
        super().check_dataset(data)     # Asserts type

        ref_shape= None                
        for channel in data.channels(): # Check each channel's shape: should be
            arr = data[channel]         #   (N, M) = (num_events, multiplicity)
                                        # for each channel.
            if arr.dtype == object:
                raise ValueError(
                    f"Channel {channel!r} is ragged. "
                    f"FixedMCodec requires fixed jet multiplicity."
                )
            if arr.ndim not in (1, 2):
                raise ValueError(
                    f"channel {channel!r} must be flattened 1D (N,) "
                    f"or 2D (N, M), got shape {arr.shape}"
                )
            if ref_shape is None:
                ref_shape = arr.shape
            elif arr.shape != ref_shape:
                raise ValueError(
                    f"Channel {channel!r} has shape {arr.shape}, "
                    f"expected {ref_shape} to match other channels"
                )

    def fit(self, data: Dataset) -> None:
        self.check_dataset(data)

        pt  = data['pt']
        eta = data["eta"]
        
        logpt = np.log1p(pt)

        self.pt_mean  = float(logpt.mean())
        self.pt_std   = float(logpt.std())
        self.eta_mean = float(eta.mean())
        self.eta_std  = float(eta.std())
                                                # Observed physical ranges 
        self.pt_min  = float(np.nanmin(pt))     # to clip samples to
        self.pt_max  = float(np.nanmax(pt))
        self.eta_min = float(np.nanmin(eta))
        self.eta_max = float(np.nanmax(eta))

    def encode(self, data: Dataset) -> NDArray:
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