from .fixedmcodec import FixedMCodec
from ..datasets import Dataset

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class MaskedMCodec(FixedMCodec):
    """Same per-channel normalisation as FixedMCodec, but padded slots (real == 0)
    are excluded from the fitted statistics and zeroed in normalised space. An
    extra `real` existence-flag channel is passed through so a fixed-M model
    can represent variable multiplicity.
    """

    s_EXPORT_KEYS = FixedMCodec.s_EXPORT_KEYS + [ "n_features" ]

    def fit(self, data: Dataset) -> None:
        self.check_dataset(data)

        real = data["real"] == 1        # Exclude padded slots from statistics
        self._fit_stats(data["pt"][real], data["eta"][real])

                                        # phi -> (sin, cos), hence + 1
        self.n_features = len(data.channels()) + 1


    def encode(self, data: Dataset) -> NDArray:
        self.check_dataset(data)

        real = data["real"].astype(np.float32)          # (n_events, M), 0/1
        encoded_channels = self._encode_channels(
            data["pt"], data["eta"], data["phi"]
        )
                                        # (n_events, M, n_features),
                                        # then flatten to
                                        # (n_events, M * n_features)
        jets = np.stack([*encoded_channels, real], axis = -1)
        jets *= real[..., None]         # Mask out fake events
        return jets.reshape(jets.shape[0], -1).astype(np.float32)


    def mask_to_weights(self, data: Dataset) -> NDArray:
        """Construct weights using `real` mask so that BDTs only learn from
        actual particles.
        """
        real = data["real"].astype(np.float32)                 # (n_events, M)
                                        # (n_events, M, n_features)
                                        # every kinematic channel weighted by 
                                        # `real`, `real` itself by 1
        weights = np.repeat(real[..., None], self.n_features, axis = -1)
        weights[..., -1] = 1.
        return weights.reshape(real.shape[0], -1) # (n_events, M * n_features)


    def decode(self, out: NDArray) -> dict[str, NDArray]:
                                        # (n_events, M * n_channels) 
                                        # -> (n_events, M, n_channels)
        jets = out.reshape(out.shape[0], -1, self.n_features)
        std_pt, std_eta, sin_phi, cos_phi, real = np.moveaxis(jets, -1, 0)

        return {
            **self._decode_channels(std_pt, std_eta, sin_phi, cos_phi),
            "real": (real > 0.5).astype(np.float32),
        }
