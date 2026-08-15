from __future__ import annotations

from .config import Config
from ..datasets import ClusteredL1Puppi
from ..codecs import PaddedCodec

from dataclasses import dataclass

#-----------------------------------------------------------------------------

@dataclass
class ClusteredL1PuppiConfig(Config):
    dataset = ClusteredL1Puppi
    codec   = PaddedCodec
