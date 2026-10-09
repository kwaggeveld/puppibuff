from __future__ import annotations

from .dataset import Dataset

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class ToyJet(Dataset):
    """Fake toy jet-level samples."""

    s_CHANNELS = [ "pt", "eta", "phi" ]

    def __init__(self, size: int = 300_000, seed: int = 0) -> None:
        self.size = size
        self.seed = seed
        super().__init__()

    def _load(self) -> dict[str, NDArray]:
        rng = np.random.default_rng(self.seed)
        noise_pt, noise_eta = rng.standard_normal((2, self.size))
                                        # Some correlation with eta
        log_pt = 2.7 + .35 * (.3 * noise_eta + np.sqrt(1 - .3**2) * noise_pt)

        return {
            "pt":  np.expm1(np.maximum(log_pt, .25)).astype(np.float32),
            "eta": np.clip(1.5 * noise_eta, -5, 5).astype(np.float32),
            "phi": rng.uniform(-np.pi, np.pi, self.size).astype(np.float32),
        }


class ToyConstituent(Dataset):
    """Fake toy constituent-level samples."""

    s_CHANNELS = [ "pt", "eta", "phi", "real" ]

    s_SLOTS = 8                         # M, slots per jet

    def __init__(self, size: int = 150_000, seed: int = 0) -> None:
        self.size = size
        self.seed = seed
        super().__init__()

    def _load(self) -> dict[str, NDArray]:
        rng = np.random.default_rng(self.seed)
        shape = (self.size, self.s_SLOTS)

        multiplicity = np.clip(rng.poisson(4, self.size), 1, self.s_SLOTS)
        real = np.arange(self.s_SLOTS) < multiplicity[:, None]

        axis_eta = rng.normal(0, 1.5, (self.size, 1))
        axis_phi = rng.uniform(-np.pi, np.pi, (self.size, 1))

        log_pt = np.maximum(rng.normal(1.5, .6, shape), .1)
        pt  = np.sort(np.expm1(log_pt), axis = 1)[:, ::-1]   # Descending
        eta = axis_eta + rng.normal(0, .1, shape)
        phi = (axis_phi + rng.normal(0, .1, shape) + np.pi) % (2 * np.pi) - np.pi

        return {
            channel: np.where(real, values, 0).astype(np.float32)
            for channel, values in
            { "pt": pt, "eta": eta, "phi": phi, "real": real }.items()
        }
