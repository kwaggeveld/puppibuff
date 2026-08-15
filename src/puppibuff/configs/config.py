from __future__ import annotations

from ..datasets import Dataset
from ..codecs import Codec
from ..build_trainds import build_trainds, Paths
from ..flowbdt import FlowBDT

from abc import ABC
from dataclasses import dataclass, field, asdict
import json

from numpy.typing import NDArray
from typing import ClassVar

#-----------------------------------------------------------------------------

DEFAULT_TREE_CONFIG = {
    "n_estimators": 50,
    "max_depth": 6,
    "objective": "reg:squarederror",
    "learning_rate": .1,
    "n_jobs": 16,
    "subsample": 1.,
    "reg_alpha": .2,
    "reg_lambda": .1,
    "seed": 666,
    "tree_method": "hist",
    "device": "cpu",
}

@dataclass
class Config(ABC):
    dataset: ClassVar[type[Dataset]]# Fixed by each concrete config
    codec:   ClassVar[type[Codec]]

    n_steps:  int = 15
    n_events: int | None = 500_000      # None => train on the entire dataset

    s1phi: bool = False                 # Encode phi -> (sin, cos) rather than
                                        # normalising it into one channel

    multi_output: bool = False          # One multi-output BDT per (step, channel)

    tree_config: dict = field(default_factory = lambda: dict(DEFAULT_TREE_CONFIG))

    def __post_init__(self) -> None:
        if self.multi_output:
            self.tree_config.setdefault("multi_strategy", "multi_output_tree")

    def setup(self, data: Dataset | None = None) -> tuple[Dataset, Codec, FlowBDT, Paths, NDArray]:
        """Load the dataset, fit + apply the codec, build the flow-matching
        training set, and construct the (untrained) model. Pass an already-loaded
        `data` to reuse it instead of reloading from disk.
        """
        data = data if data is not None else self.dataset()

        codec = self.codec(self.s1phi)
        codec.fit(data)

        x1 = codec.encode(data[:self.n_events])
        x, y = build_trainds(x1, self.n_steps)

        sizes = codec.group_sizes() if self.multi_output else None
        model = FlowBDT(self.tree_config, sizes)

        return data, codec, model, x, y

# --- Export/import ---

    def to_json(self, path: str) -> None:
        with open(path, "w") as file:
            json.dump({ "config_cls": type(self).__name__ } | asdict(self), file)

    @classmethod
    def from_json(cls, path: str) -> Config:
        from .. import configs          # Deferred to prevent circular import

        with open(path) as file:
            fields = json.load(file)

        return getattr(configs, fields.pop("config_cls"))(**fields)
