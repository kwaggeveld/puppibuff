from .fixedmcodec import FixedMCodec
from ..datasets import Dataset

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class MultiplicityCodec(FixedMCodec):
    """Same per-channel normalisation as FixedMCodec, but variable multiplicity
    is encoded to a single scalar channel instead of one `real` flag per slot.
    """

    s_EXPORT_KEYS = FixedMCodec.s_EXPORT_KEYS + [ "mult_mean", "mult_std" ]

    s_DECODED = FixedMCodec.s_DECODED + [ "real" ]

    def fit(self, data: Dataset) -> None:
        self.check_dataset(data)

        real = data["real"] == 1        # Exclude padded slots from statistics
        self._fit_stats(data["pt"][real], data["eta"][real], data["phi"][real])

        log_mult = np.log1p(data["real"].sum(axis = 1))
        self.mult_mean = float(log_mult.mean())
        self.mult_std  = float(log_mult.std())

                                        # phi -> (sin, cos) adds one extra channel
        self.n_features   = len(data.channels()) - 1 + self.s1phi   # `real` dropped
        self.multiplicity = data["real"].shape[1]   # Slots per jet, lost by encode()


    def group_sizes(self) -> list[int]:
        """One group per physical channel spanning every slot (encode() is
        channel-major, so a channel's columns are consecutive), plus the
        trailing multiplicity column on its own.
        """
        return [self.multiplicity] * self.n_features + [ 1 ]


    def encode(self, data: Dataset) -> NDArray:
        self.check_dataset(data)

        encoded_channels = self._encode_channels(
            data["pt"], data["eta"], data["phi"]
        )                               
                                        # (n_events, n_features, M)
        jets = np.stack(encoded_channels, axis = 1)

        mult     = data["real"].sum(axis = 1)
        mult_std = (np.log1p(mult) - self.mult_mean) / self.mult_std

        return np.concatenate(         # (n_events, n_features * M + 1)
            [jets.reshape(jets.shape[0], -1), mult_std[:, None]], axis = 1
        ).astype(np.float32)


    def decode(self, out: NDArray) -> dict[str, NDArray]:
                                        # (n_events, n_features * M + 1)
                                        # -> (n_events, n_features, M)
        jets = out[:, :-1].reshape(out.shape[0], self.n_features, -1)
        channels = np.moveaxis(jets, 1, 0)

        mult = np.expm1(out[:, -1] * self.mult_std + self.mult_mean)
        mult = np.clip(np.round(mult), 0, self.multiplicity).astype(int)

        real = np.arange(self.multiplicity) < mult[:, None]

        return {
            **self._decode_channels(*channels),
            "real": real.astype(np.float32),
        }


    def decode_cpp(self) -> str:
        raise NotImplementedError(
            "MultiplicityCodec has no HLS decode writer (yet)..."
        )

    def decode_params_h(self) -> str:
        raise NotImplementedError(
            "MultiplicityCodec has no HLS decode writer (yet)..."
        )

