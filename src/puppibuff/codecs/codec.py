from __future__ import annotations

from ..datasets import Dataset
from ..utils import class_path, import_class

from abc import ABC, abstractmethod
import json
from pathlib import Path

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

class Codec(ABC):
    """The interface between a `Dataset`'s physical channels and the encoded  
    space `FlowBDT` is trained in and samples from.
    """

    s_EXPORT_KEYS: list[str]            # Fitted attributes `to_json` writes
    s_DECODED: list[str]                # The channels `decode` returns

    s_DECODE_TOP = "decode"             # `decode_cpp`'s HLS top function
                                        
    s_DECODE_PARAMS = "codec_params.hh"  # The Codec's fitted constants, 
                                        # which `decode.cpp` reads

    multiplicity: int                   # Slots per event

    def __init__(self, s1phi: bool = False) -> None:  # Agrees with Config.s1phi
        self.s1phi = s1phi

# --- Main functionality --- 

    @abstractmethod
    def fit(self, data: Dataset) -> None:
        """Set every fitted constant from `data`. Called once on the whole 
        dataset, before `encode`.
        """
        ...

    @abstractmethod
    def encode(self, data: Dataset) -> NDArray:
        """Return `data`'s channels, encoded, as one `(n_events, n_columns)` 
        array. Sort the columns channel-major: channel's columns are consecutive.
        """
        ...

    @abstractmethod
    def decode(self, out: NDArray) -> dict[str, NDArray]:
        """Reverse the transformations done by `encode`, return as one physical 
        array per `s_DECODED. `out` is `(n_events, n_columns)` as `encode` 
        returns.
        """
        ...

    @abstractmethod
    def group_sizes(self) -> list[int]:
        """How many encoded columns each multi-output BDT predicts jointly.

        A group is a contiguous block of `encode`'s columns and each column must 
        be part of one group, so the widths should sum to `n_columns`. 
        `[1] * n_columns` is one single-output BDT per column.
        """
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
        """Write the Codec's class tag and `s_EXPORT_KEYS` to `path` to save the
        object to JSON.

       `from_json` uses the tag to reconstruct, so a Codec defined outside an
        importable module cannot be loaded back.
        """
        if type(self).__module__ == "__main__":
            print(f"{ type(self).__name__ } is defined in __main__, so this "
                  f"archive can only be loaded in a session that defines it.")

        with open(path, "w") as file:
            json.dump({ "codec_cls": class_path(type(self)) }
                      | { key: getattr(self, key) for key in self.s_EXPORT_KEYS }, file)

    @classmethod
    def from_json(cls, path: Path | str) -> Codec:
        """Construct the Codec specified by the file's `codec_cls` tag."""
        with open(path) as f:
            state = json.load(f)

        obj = import_class(state.pop("codec_cls"))()
        obj.__dict__.update(state)                                            # pyright: ignore[reportAttributeAccessIssue]

        return obj
