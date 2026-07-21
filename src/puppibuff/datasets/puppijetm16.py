from __future__ import annotations

from .dataset import Dataset

import numpy as np

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class PuppiJetM16(Dataset):
    s_CHANNELS = [ "pt", "eta", "phi" ]
    s_CHANNEL_KEYS = { channel: "PuppiJet_" + channel for channel in s_CHANNELS }

    def _select(self, data: dict) -> dict[str, NDArray] | None:
        mask = data["nPuppiJet"] == 16
        if not mask.any(): return None

        return {
            channel: np.stack(data[raw_key][mask])
            for channel, raw_key in self.s_CHANNEL_KEYS.items()
        }
