from __future__ import annotations

from .jetcodec import JetCodec
from .. import utils
from ..datasets import Dataset

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class ConstituentCodec(JetCodec):
    """JetCodec's per-channel normalisation for padded (N, M) constituent-level
    samples.
    """

    s_REQUIRED = JetCodec.s_REQUIRED + [ "real" ]
    s_DECODED  = JetCodec.s_DECODED  + [ "real" ]

    max_multiplicity: int               # Slots per jet, M

    def fit(self, data: Dataset) -> None:
        self._check_channels(data)
                                        # Exclude padded slots from statistics
        real = utils.bool_mask(data["real"])
        self._fit_stats(data["pt"][real], data["eta"][real], data["phi"][real])

                                        # phi -> (sin, cos) adds one
        self.n_features       = len(JetCodec.s_REQUIRED) + self.s1phi
        self.max_multiplicity = data["real"].shape[1]


    def encode(self, data: Dataset) -> NDArray:
        encoded_channels = self._encode_channels(
            data["pt"], data["eta"], data["phi"]
        )                               # (N, n_features, M)
        jets = np.stack(encoded_channels, axis = 1)
                                        # (N, n_features * M), channel-major
        return jets.reshape(len(jets), -1).astype(np.float32)


    def real_slots(self, data: Dataset) -> NDArray:
        return utils.bool_mask(data["real"])


    def decode(self, out: NDArray, multiplicities: NDArray | None = None) -> dict[str, NDArray]:
        if multiplicities is None:
            raise ValueError("ConstituentCodec requires the multiplicities that "
                             "the samples were drawn under")
                                        # (N, n_features, M) -> per channel
        jets = out.reshape(len(out), self.n_features, self.max_multiplicity)
        real = np.arange(self.max_multiplicity) < multiplicities[:, None]

        channels = self._decode_channels(*np.moveaxis(jets, 1, 0))

        return { channel: np.where(real, values, 0) for channel, values in channels.items() } \
               | { "real": real.astype(np.float32) }
