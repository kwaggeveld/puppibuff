from __future__ import annotations

from .config import Config
from ..datasets import ToyJet, ToyConstituent
from ..codecs import FixedMCodec, MultiplicityCodec

from dataclasses import dataclass

#-----------------------------------------------------------------------------

@dataclass
class ToyJetConfig(Config):
    dataset = ToyJet
    codec   = FixedMCodec


@dataclass
class ToyConstituentConfig(Config):
    dataset = ToyConstituent
    codec   = MultiplicityCodec
