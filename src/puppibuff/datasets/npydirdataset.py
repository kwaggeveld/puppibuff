from __future__ import annotations

from .dataset import Dataset

from os import environ
from pathlib import Path
import numpy as np
from tqdm import tqdm

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class NpyDirDataset(Dataset):
    """A dataset stored as a directory of `.npy` files, each holding a pickled
    dict of raw branch arrays.

    The files are concatenated per channel and narrowed per file by `_select`.
    `dir` defaults to `s_LOCATION_ENV`'s environment variable's content.
    """

    s_CHANNEL_KEYS: dict[str, str]      # Maps channel -> raw key, for `_select`
    s_LOCATION_ENV: str                 # Environ var pointing at the .npy dir

    def __init__(self, dir: str | None = None) -> None:
        self.dir = dir if dir is not None else environ[self.s_LOCATION_ENV]
        super().__init__()

    def _load(self) -> dict[str, NDArray]:
        event_dict: dict[str, list[NDArray]] = { channel: [] for channel in self.s_CHANNELS }

        files = sorted(Path(self.dir).glob("*.npy"))
        for file in tqdm(files, desc = "Loading dataset"):
            data = np.load(file, allow_pickle = True).item()

            for channel, arr in self._select(data).items():
                event_dict[channel].append(arr)

        return {                        # Concat. all events per channel
            channel: np.concatenate(event_dict[channel], dtype = np.float32)
            for channel in self.s_CHANNELS
        }

    def _select(self, data: dict) -> dict[str, NDArray]:
        """Select this dataset's events and channels out of one loaded batch"""
        return data
