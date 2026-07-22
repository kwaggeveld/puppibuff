from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from os import environ
from typing import overload

import numpy as np
from tqdm import tqdm

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class Dataset(ABC):
    s_CHANNELS: list[str]
    s_CHANNEL_KEYS: dict[str, str]      # maps channel -> raw key
    s_LOCATION_ENV: str                 # Environ var pointing at this dataset's .npy dir

    def __init__(self, dir: str | None = None) -> None:
        dir = dir if dir is not None else environ[self.s_LOCATION_ENV]
        self.d_data = self._load(dir)

# --- Loading data ---
    
    def _load(self, dir: str) -> dict[str, NDArray]:
        event_dict: dict[str, list[NDArray]] = { channel: [] for channel in self.s_CHANNELS }

        files = sorted(Path(dir).glob("*.npy"))
        for file in tqdm(files, desc = "Loading dataset"):
            data = np.load(file, allow_pickle = True).item()

            selected = self._select(data)    # Derived datasets implement this
            if selected is None: continue    # to specify their events

            for channel, arr in selected.items():
                event_dict[channel].append(arr) 
                                        
        return {                        # Concat. all events per channel
            channel: np.concatenate(event_dict[channel], dtype = np.float32)
            for channel in self.s_CHANNELS
        }

    @abstractmethod                     # Selected events from one file
    def _select(self, data: dict) -> dict[str, NDArray] | None:
        ...                         

# --- Accessors ---

    @overload
    def __getitem__(self, key: str) -> NDArray: ...
    @overload
    def __getitem__(self, key: slice) -> Dataset: ...

    def __getitem__(self, key: str | slice) -> NDArray | Dataset:
        if isinstance(key, str):        # string key -> return features
            return self.d_data[key]
                                        # slice key -> return sliced Dataset
        obj = object.__new__(type(self))
        obj.d_data = { channel: arr[key] for channel, arr in self.d_data.items() }
        return obj

    def channels(self) -> list[str]:
        return list(self.s_CHANNELS)
