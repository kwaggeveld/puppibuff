from __future__ import annotations

from .config import Config
from ..datasets import FlatPuppiJet
from ..codecs import JetCodec

from dataclasses import dataclass

#-----------------------------------------------------------------------------

@dataclass
class FlatPuppiJetConfig(Config):
    dataset = FlatPuppiJet
    codec   = JetCodec
