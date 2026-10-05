"""DIEP on PyTorch Geometric -- a standalone package.

This package has no parent and no DGL dependency: everything it needs is vendored here,
including the framework-neutral pieces (MLP blocks, activations, cutoffs, grid geometry,
checkpoint IO) that used to be imported from the DGL-era ``diep``. Parameter names still
match that implementation, so compatible checkpoints load directly. The PyG model
additionally applies two-body cutoff smoothing, so its predictions differ from the
unsmoothed DGL backend even with the same weights.

Layout::

    diep_pyg.graph.compute      bond vectors, edge pruning, the three-body line graph
    diep_pyg.graph.converters   pymatgen Structure / Molecule -> DIEPData
    diep_pyg.graph.data         dataset, collate functions and dataloaders
    diep_pyg.layers             embedding, three-body, graph convolution, readouts
    diep_pyg.models             the DIEP model
    diep_pyg.apps.pes           Potential (energies, forces, stresses, hessian)
    diep_pyg.utils.training     Lightning modules

    diep_pyg.matpes             MatPES -> graph cache, fold splits, train/val/test loaders
    diep_pyg.train              the training script that produced diep_fold1
    diep_pyg.export             training checkpoint -> self-contained model directory
    diep_pyg.evaluate           metrics and per-structure predictions on a split
    diep_pyg.pretrained         load_potential / make_calculator for an exported model

The three-body line graph is built directly in *parent-bond index space*: the triple index
holds ids of bonds in the full graph, never of the pruned subset. That is the invariant the
DGL side had to reach by remapping, and :func:`diep_pyg.graph.compute.assert_lg_invariants`
checks it here.
"""

from __future__ import annotations

from diep_pyg.graph.compute import (
    DIEPData,
    assert_lg_invariants,
    compute_pair_vector_and_distance,
    create_line_graph,
    prune_edges_by_features,
)
from diep_pyg.graph.converters import Molecule2Graph, Structure2Graph, get_element_list

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
