"""Layers for the PyG backend."""

from __future__ import annotations

from diep.layers._atom_ref import AtomRef
from diep.layers._diep import DIEPIntegrator
from diep.layers._embedding import EmbeddingBlock
from diep.layers._graph_convolution import M3GNetBlock, M3GNetGraphConv
from diep.layers._readout import (
    ReduceReadOut,
    Set2SetReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from diep.layers._three_body import ThreeBodyInteractions
from diep.layers._zbl import NuclearRepulsion

__all__ = [
    "AtomRef",
    "DIEPIntegrator",
    "EmbeddingBlock",
    "M3GNetBlock",
    "M3GNetGraphConv",
    "NuclearRepulsion",
    "ReduceReadOut",
    "Set2SetReadOut",
    "ThreeBodyInteractions",
    "WeightedAtomReadOut",
    "WeightedReadOut",
]
