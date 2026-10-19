from __future__ import annotations

from .config import Config
from ..datasets import L1Puppi
from ..codecs import ConditionedConstituentCodec, ConditionedJetCodec, ConstituentCodec

from dataclasses import dataclass

#-----------------------------------------------------------------------------

@dataclass
class L1PuppiJetConfig(Config):
    dataset = L1Puppi
    codec   = ConditionedJetCodec


@dataclass
class L1PuppiConfig(Config):
    dataset = L1Puppi
    codec   = ConstituentCodec


@dataclass
class ConditionedL1PuppiConfig(Config):
    dataset = L1Puppi
    codec   = ConditionedConstituentCodec
