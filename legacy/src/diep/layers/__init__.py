"""This package implements the layers for different Graph Neural Networks.

Imports are lazy (PEP 562). Several layers in this package are implemented on DGL, which is
not installable everywhere (there is no linux-aarch64 wheel, for instance). Resolving names
on demand means ``diep.layers.MLP`` and the other framework-agnostic pieces -- which the
PyG backend in :mod:`diep.pyg` reuses -- stay importable when DGL is absent, while
``diep.layers.M3GNetBlock`` and friends still raise a clear ImportError if you ask for them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_MODULE_BY_NAME = {
    "ActivationFunction": "diep.layers._activations",
    "AtomRef": "diep.layers._atom_ref",
    "FourierExpansion": "diep.layers._basis",
    "RadialBesselFunction": "diep.layers._basis",
    "BondExpansion": "diep.layers._bond",
    "MLP": "diep.layers._core",
    "EdgeSet2Set": "diep.layers._core",
    "GatedEquivariantBlock": "diep.layers._core",
    "GatedMLP": "diep.layers._core",
    "MLP_norm": "diep.layers._core",
    "build_gated_equivariant_mlp": "diep.layers._core",
    "DIEPIntegrator": "diep.layers._diep",
    "EmbeddingBlock": "diep.layers._embedding",
    "M3GNetBlock": "diep.layers._graph_convolution",
    "M3GNetGraphConv": "diep.layers._graph_convolution",
    "GraphNorm": "diep.layers._norm",
    "AttentiveFPReadout": "diep.layers._readout",
    "GlobalPool": "diep.layers._readout",
    "ReduceReadOut": "diep.layers._readout",
    "Set2SetReadOut": "diep.layers._readout",
    "WeightedAtomReadOut": "diep.layers._readout",
    "WeightedReadOut": "diep.layers._readout",
    "WeightedReadOutPair": "diep.layers._readout",
    "ThreeBodyInteractions": "diep.layers._three_body",
    "NuclearRepulsion": "diep.layers._zbl",
}

__all__ = sorted(_MODULE_BY_NAME)


def __getattr__(name: str):
    """Resolve a layer name to its defining module on first access."""
    try:
        module_name = _MODULE_BY_NAME[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    import importlib

    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return __all__


if TYPE_CHECKING:  # keep static analysis and IDEs working
    from diep.layers._activations import ActivationFunction
    from diep.layers._atom_ref import AtomRef
    from diep.layers._basis import FourierExpansion, RadialBesselFunction
    from diep.layers._bond import BondExpansion
    from diep.layers._core import (
        MLP,
        EdgeSet2Set,
        GatedEquivariantBlock,
        GatedMLP,
        MLP_norm,
        build_gated_equivariant_mlp,
    )
    from diep.layers._diep import DIEPIntegrator
    from diep.layers._embedding import EmbeddingBlock
    from diep.layers._graph_convolution import M3GNetBlock, M3GNetGraphConv
    from diep.layers._norm import GraphNorm
    from diep.layers._readout import (
        AttentiveFPReadout,
        GlobalPool,
        ReduceReadOut,
        Set2SetReadOut,
        WeightedAtomReadOut,
        WeightedReadOut,
        WeightedReadOutPair,
    )
    from diep.layers._three_body import ThreeBodyInteractions
    from diep.layers._zbl import NuclearRepulsion
