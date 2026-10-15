from __future__ import annotations

from .config import Config
from ..datasets import ClusteredL1Puppi, RelativeL1Puppi
from ..codecs import ConstituentCodec

from dataclasses import dataclass

#-----------------------------------------------------------------------------

@dataclass
class ClusteredL1PuppiConfig(Config):
    dataset = ClusteredL1Puppi
    codec   = ConstituentCodec


@dataclass
class RelativeL1PuppiConfig(Config):
    dataset = RelativeL1Puppi
    codec   = ConstituentCodec
