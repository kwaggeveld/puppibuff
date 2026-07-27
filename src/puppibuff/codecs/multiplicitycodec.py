from .fixedmcodec import FixedMCodec
from ..datasets import Dataset

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class MultiplicityCodec(FixedMCodec):
    """Same per-channel normalisation as FixedMCodec, but variable multiplicity
    is encoded to a single scalar channel instead of one `real` flag per slot.
    """

    s_EXPORT_KEYS = (FixedMCodec.s_EXPORT_KEYS 
                        + [ "n_features", "multiplicity", 
                            "mult_mean", "mult_std", "mult_cdf" ])

    def fit(self, data: Dataset) -> None:
        self.check_dataset(data)

        real = data["real"] == 1        # Exclude padded slots from statistics
        self._fit_stats(data["pt"][real], data["eta"][real], data["phi"][real])

                                        # phi -> (sin, cos) adds one extra channel
        self.n_features   = len(data.channels()) - 1 + self.s1phi   # `real` dropped
        self.multiplicity = data["real"].shape[1]   # Slots per jet, lost by encode()

        mult = data["real"].sum(axis = 1)
        self.mult_mean = float(mult.mean())
        self.mult_std  = float(mult.std())
                                        # Empirical CDF over 0 ... M, so decode()
                                        # can quantile-match instead of rounding
        counts = np.bincount(mult.astype(int), minlength = self.multiplicity + 1)
        self.mult_cdf = (np.cumsum(counts) / len(mult)).tolist()


    def group_sizes(self) -> list[int]:
        """One group per physical channel spanning every slot (encode() is
        channel-major, so a channel's columns are consecutive), plus the
        trailing multiplicity column on its own.
        """
        return [self.multiplicity] * self.n_features + [ 1 ]


    def encode(self, data: Dataset) -> NDArray:
        self.check_dataset(data)

        real = data["real"].astype(np.float32)          # (n_events, M), 0/1
        encoded_channels = self._encode_channels(
            data["pt"], data["eta"], data["phi"]
        )                               
                                        # (n_events, n_features, M)
        jets = np.stack(encoded_channels, axis = 1)

        mult = (real.sum(axis = 1) - self.mult_mean) / self.mult_std

        return np.concatenate(         # (n_events, n_features * M + 1)
            [jets.reshape(jets.shape[0], -1), mult[:, None]], axis = 1
        ).astype(np.float32)


    def decode(self, out: NDArray) -> dict[str, NDArray]:
                                        # (n_events, n_features * M + 1)
                                        # -> (n_events, n_features, M)
        jets = out[:, :-1].reshape(out.shape[0], self.n_features, -1)
        channels = np.moveaxis(jets, 1, 0)

        mult = self._decode_multiplicity(out[:, -1])
        real = np.arange(self.multiplicity) < mult[:, None]

        return {
            **self._decode_channels(*channels),
            "real": real.astype(np.float32),
        }


    def _decode_multiplicity(self, out: NDArray) -> NDArray:
        """Quantile-match `out` onto the fitted multiplicity distribution.
        Map output multiplicity distribution onto observed cdf exactly.
        """
        ranks     = np.argsort(np.argsort(out))       # each jet's rank among all predictions
        quantiles = (ranks + .5) / len(out)           # rank -> quantile in (0, 1)
        return np.searchsorted(self.mult_cdf, quantiles)  # quantile -> integer bin
