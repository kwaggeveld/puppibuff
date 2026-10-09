from __future__ import annotations

from .config import Config
from ..datasets import ToyJet, ToyConstituent
from ..codecs import JetCodec, MultiplicityCodec

from dataclasses import dataclass

#-----------------------------------------------------------------------------

@dataclass
class ToyJetConfig(Config):
    dataset = ToyJet
    codec   = JetCodec


@dataclass
class ToyConstituentConfig(Config):
    dataset = ToyConstituent
    codec   = MultiplicityCodec
