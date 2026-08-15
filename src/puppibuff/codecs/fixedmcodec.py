from .codec import Codec
from ..datasets import Dataset
from ..utils import fill_template

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class FixedMCodec(Codec):
    """Per-channel codec for fixed multiplicity M (pt, eta, phi) events.

    pt  -> log1p -> normalise (over all jets)
    eta -> normalise          (over all jets)
    phi -> (sin_phi, cos_phi), or plain normalise if `s1phi = False`
    """

    s_EXPORT_KEYS = [ channel + "_" + attr
                      for channel in ( "pt", "eta", "phi" )
                      for attr    in ( "mean", "std", "min", "max" )] \
                    + [ "s1phi", "n_features", "multiplicity" ]

    s_DECODED = [ "pt", "eta", "phi" ]  # What `decode` returns, per slot

    s_FRACTION_BITS = 12                # Decoded outputs' fractional precision

    s_EXPM1_FRAC_BITS = 8               # `expm1` table entries per unit of log1p(pt)


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

        self._fit_stats(data["pt"], data["eta"], data["phi"])

        self.n_features   = len(data.channels()) + self.s1phi   # phi -> (sin, cos) adds one
        self.multiplicity = data["pt"].shape[1] if data["pt"].ndim == 2 else 1

    def encode(self, data: Dataset) -> NDArray:
        self.check_dataset(data)

        encoded_channels = self._encode_channels(
            data["pt"], data["eta"], data["phi"]
        )
                                        # from 4 x (N, M) to (N, 4, M)
        return np.stack([*encoded_channels], axis = 1)


    def decode(self, out: NDArray) -> dict[str, NDArray]:
        encoded_channels = np.moveaxis(out, 1, 0)
        return self._decode_channels(*encoded_channels)


    def group_sizes(self) -> list[int]:
        """One block of size `M` for each feature"""
        return [self.multiplicity] * self.n_features


    @property
    def n_decoded(self) -> int:
        """Every decoded channel, for every slot."""
        return len(self.s_DECODED) * self.multiplicity


    @property
    def decoded_precision(self) -> str:
        """An `ap_fixed` covering every decoded channel's observed range."""
        largest = max(abs(self.pt_min),  abs(self.pt_max),
                      abs(self.eta_min), abs(self.eta_max),
                      np.pi)            # phi wraps to [-pi, pi)

        integer = int(np.ceil(np.log2(largest))) + 1        # Plus a sign bit

        return f"ap_fixed<{ integer + self.s_FRACTION_BITS }, { integer }>"


    @property
    def expm1_scaling(self) -> int:
        """`expm1` table entries per unit of log1p(pt). Used to convert value 
        <-> table idx. idx / scale in the init, value * scale in the lookup.
        """
        return 1 << self.s_EXPM1_FRAC_BITS


    @property
    def expm1_table_size(self) -> int:
        """Enough `expm1` entries to reach log1p(pt_max) given the scaling,
        so that clamping the index saturates at the observed range.
        """
        return int(np.ceil(np.log1p(self.pt_max) * self.expm1_scaling))


    @property
    def expm1_table(self) -> str:
        """Compute the `expm1` table's entries, format as C++ initialiser list."""
                                        # Bottom clipped to pt_min, the top is
        entries = np.maximum(           # within a bin of pt_max by construction
            np.expm1(np.arange(self.expm1_table_size) / self.expm1_scaling),
            self.pt_min,
        )

        per_line = 6
        return "\n".join(
            "    " + " ".join(f"{ entry :.6f}," for entry in row)
            for row in (entries[idx : idx + per_line]
                        for idx in range(0, len(entries), per_line))
        )


    def decode_cpp(self) -> str:
        if self.s1phi:                  # `_decode_channels` needs arctan2
            raise NotImplementedError(
                "`decode_cpp` cannot emit the s1phi (sin, cos) decoding of phi. "
                "Fit the codec with s1phi = False."
            )

        return fill_template("puppibuff.codecs", "fixedmcodec.cpp")


    def decode_params_hh(self) -> str:
        return fill_template("puppibuff.codecs", self.s_DECODE_PARAMS,
            multiplicity = self.multiplicity,

            pt_mean  = self.pt_mean,
            pt_std   = self.pt_std,
            eta_mean = self.eta_mean,
            eta_std  = self.eta_std,
            phi_mean = self.phi_mean / (2 * np.pi),  # phi is decoded in turns
            phi_std  = self.phi_std  / (2 * np.pi),

            eta_min = self.eta_min,
            eta_max = self.eta_max,
            two_pi  = 2 * np.pi,

            expm1_scaling    = self.expm1_scaling,
            expm1_table_size = self.expm1_table_size,
            expm1_table      = self.expm1_table,

            turns_width = self.s_FRACTION_BITS + 4,
        )


    def _fit_stats(self, pt: NDArray, eta: NDArray, phi: NDArray) -> None:
        """Set pt/eta/phi mean/std/min/max from arrays."""
        logpt = np.log1p(pt)

        self.pt_mean  = float(logpt.mean())
        self.pt_std   = float(logpt.std())
        self.eta_mean = float(eta.mean())
        self.eta_std  = float(eta.std())
        self.phi_mean = float(phi.mean())       # Unused when self.s1phi == True
        self.phi_std  = float(phi.std())
                                                
        self.pt_min  = float(np.nanmin(pt))     # Observed physical ranges 
        self.pt_max  = float(np.nanmax(pt))     # to clip samples to
        self.eta_min = float(np.nanmin(eta))
        self.eta_max = float(np.nanmax(eta))
        self.phi_min = float(np.nanmin(phi))
        self.phi_max = float(np.nanmax(phi))


    def _encode_channels(
        self, pt: NDArray, eta: NDArray, phi: NDArray
    ) -> tuple[NDArray, ...]:
        """Normalise log(1 + pt) and eta. Encode phi as (sin(phi), cos(phi)) 
        if self.s1phi, else normalise.
        """
        std_pt  = (np.log1p(pt) - self.pt_mean) / self.pt_std
        std_eta = (eta - self.eta_mean) / self.eta_std

        if self.s1phi:
            return std_pt, std_eta, np.sin(phi), np.cos(phi)

        std_phi = (phi - self.phi_mean) / self.phi_std
        return std_pt, std_eta, std_phi


    def _decode_channels(                      # (sin, cos) or std_phi
        self, std_pt: NDArray, std_eta: NDArray, *phi_channels: NDArray
    ) -> dict[str, NDArray]:
        """Rescale e^pt - 1 and eta back to observed mean, std. Decode phi
        as arctan2(sin(phi), cos(phi)) if self.s1phi, else rescale back, then
        wrap to [-pi, pi).
        """
        pt  = np.expm1(std_pt * self.pt_std + self.pt_mean)
        eta = std_eta * self.eta_std + self.eta_mean

                                        # if self.s1phi: 
                                        #   phi_channels = (sin_phi, cos_phi)
                                        # else:     phi_channels = (std_phi,)
        phi = (np.arctan2(*phi_channels) if self.s1phi          # -> (-pi, pi]
                else phi_channels[0] * self.phi_std + self.phi_mean)
        
        return {                        # Clip to observed range, 
            "pt":  np.clip(pt,  self.pt_min,  self.pt_max),
            "eta": np.clip(eta, self.eta_min, self.eta_max),
            "phi": (phi + np.pi) % (2 * np.pi) - np.pi,
        }                               # wrap phi to [-pi, pi)