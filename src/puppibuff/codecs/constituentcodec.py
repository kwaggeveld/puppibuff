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

    s_CONDITIONS = [ "multiplicity" ]   # Multiplicity first: it masks the slots

    s_REQUIRED = JetCodec.s_REQUIRED + [ "real" ] + s_CONDITIONS
    s_DECODED  = JetCodec.s_DECODED  + [ "real" ] + s_CONDITIONS

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


    def decode(self, out: NDArray) -> dict[str, NDArray]:
                                        # (N, n_features * M + n_conditions),
                                        # conditions last -> (N, n_features, M)
        n_encoded  = self.n_features * self.max_multiplicity
        jets       = out[:, :n_encoded].reshape(len(out), self.n_features, self.max_multiplicity)
        conditions = out[:, n_encoded:]
        real       = utils.real_slots(conditions[:, 0], self.max_multiplicity)

        channels = zip(self.s_KINEMATICS, self._decode_channels(*np.moveaxis(jets, 1, 0)))

        return { channel: np.where(real, values, 0) for channel, values in channels } \
               | { "real": real.astype(np.float32) } \
               | dict(zip(self.s_CONDITIONS, conditions.T))


class ConditionedConstituentCodec(ConstituentCodec):
    """ConstituentCodec whose conditions include the jet-level kinematics."""

    s_CONDITIONS = ConstituentCodec.s_CONDITIONS + [ "jet_pt", "jet_eta", "jet_phi" ]

    s_REQUIRED = JetCodec.s_REQUIRED + [ "real" ] + s_CONDITIONS
    s_DECODED  = JetCodec.s_DECODED  + [ "real" ] + s_CONDITIONS
