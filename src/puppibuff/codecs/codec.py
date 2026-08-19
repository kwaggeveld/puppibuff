from __future__ import annotations

from ..datasets import Dataset

from abc import ABC, abstractmethod
import json
from pathlib import Path

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class Codec(ABC):
    s_EXPORT_KEYS: list[str]
    s_DECODED: list[str]                # The channels `decode` returns

    s_DECODE_TOP = "decode"             # `decode_cpp`'s HLS top function
                                        
    s_DECODE_PARAMS = "codec_params.hh"  # The Codec's fitted constants, 
                                        # which `decode.cpp` reads

    multiplicity: int                   # Slots per event

    def __init__(self, s1phi: bool = False) -> None:  # Agrees with Config.s1phi
        self.s1phi = s1phi

    def check_dataset(self, data: Dataset) -> None:
        if not isinstance(data, Dataset):
            raise TypeError(f"expected a Dataset, got {type(data).__name__}")

# --- Main functionality --- 

    @abstractmethod
    def fit(self, data: Dataset) -> None:
        ...

    @abstractmethod
    def encode(self, data: Dataset) -> NDArray:
        ...

    @abstractmethod
    def decode(self, out: NDArray) -> dict[str, NDArray]:
        ...

    @abstractmethod
    def group_sizes(self) -> list[int]:
        ...

# --- HLS export ---

    @property
    @abstractmethod
    def n_decoded(self) -> int:
        """How many values `decode` returns per event, i.e. how many output
        links the FPGA design needs.
        """
        ...

    @property
    @abstractmethod
    def decoded_precision(self) -> str:
        """The `ap_fixed` type for decoded events"""
        ...

    @abstractmethod
    def decode_cpp(self) -> str:
        """Write `firmware/decode.cpp`: the HLS block that decodes sampled events
        from normalised space.
        """
        ...

    @abstractmethod
    def decode_params_hh(self) -> str:
        """Write `firmware/codec_params.hh`, the fitted constants `decode.cpp`
        reads.
        """
        ...

# --- Export/import ---

    def to_json(self, path: Path | str) -> None:
        with open(path, "w") as file:
            json.dump({ "codec_cls": type(self).__name__ }
                      | { key: getattr(self, key) for key in self.s_EXPORT_KEYS }, file)

    @classmethod
    def from_json(cls, path: Path | str) -> Codec:
        from .. import codecs            # Deferred to prevent circular import

        with open(path) as f:
            state = json.load(f)

        obj = getattr(codecs, state.pop("codec_cls"))()
        obj.__dict__.update(state)                                            # pyright: ignore[reportAttributeAccessIssue]

        return obj
