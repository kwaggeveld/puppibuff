from ..datasets import Dataset

from abc import ABC, abstractmethod
import json

from numpy.typing import NDArray


class Codec(ABC):
    s_REQUIRED_CHANNELS: list[str]
    s_KEYS: list[str]

    def check_dataset(self, data: Dataset) -> None:
        if not isinstance(data, Dataset):
            raise TypeError(f"expected a Dataset, got {type(data).__name__}")

        missing = [c for c in self.s_REQUIRED_CHANNELS if c not in data.channels()]
        if missing:
            raise ValueError(
                f"{type(data).__name__} is missing required channels: {missing}"
            )
       
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

# --- Export/import ---

    def to_json(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump({key: getattr(self, key) for key in self.s_KEYS}, file)

    @classmethod
    def from_json(cls, path: str) -> Codec:
        obj = cls()
        with open(path) as f:
            obj.__dict__.update(json.load(f))                                 # pyright: ignore[reportAttributeAccessIssue]

        return obj
