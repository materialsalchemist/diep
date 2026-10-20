"""diep 1.0: DIEP on PyTorch Geometric -- a standalone package.

This package has no parent and no DGL dependency: everything it needs is vendored here,
including the framework-neutral pieces (MLP blocks, activations, cutoffs, grid geometry,
checkpoint IO) that used to be imported from the DGL-era ``diep`` 0.x. Parameter names still
match that implementation, so compatible checkpoints load directly. The PyG model
additionally applies two-body cutoff smoothing, so its predictions differ from the
unsmoothed DGL backend even with the same weights.

Layout::

    diep.graph.compute      bond vectors, edge pruning, the three-body line graph
    diep.graph.converters   pymatgen Structure / Molecule -> DIEPData
    diep.graph.data         dataset, collate functions and dataloaders
    diep.layers             embedding, three-body, graph convolution, readouts
    diep.models             the DIEP model
    diep.apps.pes           Potential (energies, forces, stresses, hessian)
    diep.utils.training     Lightning modules

    diep.matpes             MatPES -> graph cache, fold splits, train/val/test loaders
    diep.train              the training script that produced diep_fold1
    diep.export             training checkpoint -> self-contained model directory
    diep.evaluate           metrics and per-structure predictions on a split
    diep.pretrained         load_potential / make_calculator for an exported model

The three-body line graph is built directly in *parent-bond index space*: the triple index
holds ids of bonds in the full graph, never of the pruned subset. That is the invariant the
DGL side had to reach by remapping, and :func:`diep.graph.compute.assert_lg_invariants`
checks it here.
"""

from __future__ import annotations

__version__ = "1.0.0"

from diep.graph.compute import (
    DIEPData,
    assert_lg_invariants,
    compute_pair_vector_and_distance,
    create_line_graph,
    prune_edges_by_features,
)
from diep.graph.converters import Molecule2Graph, Structure2Graph, get_element_list

__all__ = [
    "__version__",
    "DIEPData",
    "Molecule2Graph",
    "Structure2Graph",
    "assert_lg_invariants",
    "compute_pair_vector_and_distance",
    "create_line_graph",
    "get_element_list",
    "prune_edges_by_features",
]
