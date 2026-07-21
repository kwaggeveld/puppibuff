from __future__ import annotations

from ..datasets import Dataset
from ..codecs import Codec

from abc import ABC
from dataclasses import dataclass, field
from typing import ClassVar

#-----------------------------------------------------------------------------

DEFAULT_TREE_CONFIG: dict = {           # Config copied from BUFF .ipynb
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
    dataset_cls: ClassVar[type[Dataset]]# Fixed by each concrete config
    codec_cls:   ClassVar[type[Codec]]

    n_steps:  int = 15
    n_events: int | None = 500_000      # None => train on the entire dataset

    tree_config: dict = field(default_factory = lambda: dict(DEFAULT_TREE_CONFIG))
