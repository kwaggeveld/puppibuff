from __future__ import annotations

from .config import Config
from ..datasets import ClusteredL1Puppi
from ..codecs import MultiplicityCodec

from dataclasses import dataclass

#-----------------------------------------------------------------------------

@dataclass
class MultiplicityL1PuppiConfig(Config):
    dataset = ClusteredL1Puppi
    codec   = MultiplicityCodec
