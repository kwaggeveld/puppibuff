from __future__ import annotations

from .config import Config
from ..datasets import ToyJet, ToyConstituent
from ..codecs import ConstituentCodec, JetCodec

from dataclasses import dataclass

#-----------------------------------------------------------------------------

@dataclass
class ToyJetConfig(Config):
    dataset = ToyJet
    codec   = JetCodec


@dataclass
class ToyConstituentConfig(Config):
    dataset = ToyConstituent
    codec   = ConstituentCodec
