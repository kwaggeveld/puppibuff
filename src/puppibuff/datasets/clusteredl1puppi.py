from .dataset import Dataset

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class ClusteredL1Puppi(Dataset):
    """Pre-clustered, zero-padded L1Puppi jet constituents (CL1P).

    Clustering + padding are done offline: each .npy file holds dense
    (n_jets, m) arrays per channel (jets padded/truncated to a fixed m), plus a
    `real` mask (1 = genuine constituent, 0 = padding). This class only loads
    them — no FastJet/awkward at runtime.
    """

    s_CHANNELS = [ "pt", "eta", "phi", "real" ]
    s_CHANNEL_KEYS = { channel: channel for channel in s_CHANNELS }   # Unused
    s_LOCATION_ENV = "CLUSTERED_L1PUPPI_LOCATION"

    def _select(self, data: dict) -> dict[str, NDArray]:
        return data                     # already dense (n_jets, m) per channel


