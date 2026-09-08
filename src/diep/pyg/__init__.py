"""PyTorch Geometric backend for DIEP.

An alternative to the DGL implementation in the rest of :mod:`diep`, with the same model,
the same parameter names (so DGL checkpoints load directly) and the same numbers -- but no
DGL dependency, which matters wherever DGL has no wheel (linux-aarch64, recent PyTorch).

Layout mirrors the DGL side::

    diep.pyg.graph.compute      bond vectors, edge pruning, the three-body line graph
    diep.pyg.graph.converters   pymatgen Structure / Molecule -> DIEPData
    diep.pyg.graph.data         dataset, collate functions and dataloaders
    diep.pyg.layers             embedding, three-body, graph convolution, readouts
    diep.pyg.models             the DIEP model
    diep.pyg.apps.pes           Potential (energies, forces, stresses, hessian)
    diep.pyg.utils.training     Lightning modules

The three-body line graph is built directly in *parent-bond index space*: the triple index
holds ids of bonds in the full graph, never of the pruned subset. That is the same invariant
the DGL side enforces via ``_remap_line_graph_to_bond_space``, and
:func:`diep.pyg.graph.compute.assert_lg_invariants` checks it here too.
"""

from __future__ import annotations

from diep.pyg.graph.compute import (
    DIEPData,
    assert_lg_invariants,
    compute_pair_vector_and_distance,
    create_line_graph,
    prune_edges_by_features,
)
from diep.pyg.graph.converters import Molecule2Graph, Structure2Graph, get_element_list

__all__ = [
    "DIEPData",
    "Molecule2Graph",
    "Structure2Graph",
    "assert_lg_invariants",
    "compute_pair_vector_and_distance",
    "create_line_graph",
    "get_element_list",
    "prune_edges_by_features",
]
