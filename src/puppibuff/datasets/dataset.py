from __future__ import annotations

from abc import ABC, abstractmethod
from typing import overload

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class Dataset(ABC):
    """One physics dataset as per-channel arrays, loaded once at construction.

    Override `_load` to load your own data, see its docstring.
    """

    s_CHANNELS: list[str]               # The channels `_load` must return

    def __init__(self) -> None:
        self.d_data = self._load()

        if set(self.d_data) != set(self.s_CHANNELS):
            raise ValueError(f"{ type(self).__name__ }._load returned channels "
                             f"{ sorted(self.d_data) }, expected "
                             f"{ sorted(self.s_CHANNELS) }.")

# --- Loading data ---

    @abstractmethod
    def _load(self) -> dict[str, NDArray]:
        """Return this dataset's channels as an array per name in `s_CHANNELS`.

        The only method a Dataset has to implement: load the arrays and hand them
        back.
        """
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
        """The channel names stored in the Dataset.
        """
        return list(self.s_CHANNELS)
