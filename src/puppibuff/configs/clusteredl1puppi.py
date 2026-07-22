from __future__ import annotations

from .config import Config
from ..datasets import ClusteredL1Puppi
from ..codecs import MaskedMCodec

from dataclasses import dataclass

#-----------------------------------------------------------------------------

@dataclass
class ClusteredL1PuppiConfig(Config):
    dataset_cls = ClusteredL1Puppi
    codec_cls   = MaskedMCodec
