from __future__ import annotations

from .datasets import Dataset
from .codecs import Codec
from .build_trainds import build_trainds, Paths
from .flowbdt import FlowBDT
from .configs import Config

from numpy.typing import NDArray

#-----------------------------------------------------------------------------

def setup_from_config(config: Config) -> tuple[Dataset, Codec, FlowBDT, Paths, NDArray]:
    """Load the dataset, fit + apply the codec, build the flow-matching
    training set, and construct the (untrained) model using the config."""
    data = config.dataset_cls()

    codec = config.codec_cls(config.s1phi)
    codec.fit(data)

    x1 = codec.encode(data[:config.n_events])
    x, y = build_trainds(x1, config.n_steps)

    sizes = codec.group_sizes() if config.multi_output else None
    model = FlowBDT(config.tree_config, sizes)

    return data, codec, model, x, y
