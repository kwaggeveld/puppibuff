from .dataset import Dataset

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class FlatPuppiJet(Dataset):
    s_CHANNELS = [ "pt", "eta", "phi" ]
    s_CHANNEL_KEYS = { channel: "PuppiJet_" + channel for channel in s_CHANNELS }

    def _select(self, data: dict) -> dict[str, NDArray]:
        return {
            channel: np.concatenate(data[raw_key])
            for channel, raw_key in self.s_CHANNEL_KEYS.items()
        }
