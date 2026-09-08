"""The three-body line graph on PyG graphs lives in parent-bond index space.

Same invariants the DGL side enforces (see tests/graph/test_line_graph_index_space.py), but
checked against the PyG construction, which builds the triple index in parent-bond space
directly rather than remapping out of a pruned graph.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from diep.pyg.graph.compute import (
    DIEPData,
    _triples_from_bonds,
    assert_lg_invariants,
    compute_theta,
    create_line_graph,
    ensure_line_graph_compatibility,
    prune_edges_by_features,
)
from torch_geometric.data import Batch

from .conftest import CUTOFF, THREEBODY_CUTOFF, build_graph


def test_line_graph_is_in_parent_bond_index_space(pruning_graph):
    """The whole invariant set: sizing, ordering and n_triple_ij."""
    g = pruning_graph
    assert_lg_invariants(g)
    assert int(g.n_triple_ij.numel()) == g.edge_index.size(1)
    assert int(g.triple_index.max()) < g.edge_index.size(1)


def test_pruning_actually_happens_for_the_shipped_cutoffs(pruning_graph):
    """Guard the guard: with nothing pruned the two index spaces coincide and the tests
    below would pass vacuously."""
    g = pruning_graph
    n_kept = int((g.bond_dist <= THREEBODY_CUTOFF).sum())
    assert n_kept < g.edge_index.size(1)


def test_no_triple_references_a_bond_beyond_the_threebody_cutoff(pruning_graph):
    g = pruning_graph
    first, second = g.triple_index
    assert int((g.bond_dist[first] > THREEBODY_CUTOFF).sum()) == 0
    assert int((g.bond_dist[second] > THREEBODY_CUTOFF).sum()) == 0


def test_every_triple_shares_a_centre_atom(pruning_graph):
    """The defining property of a three-body triple: both bonds start at the same atom."""
    g = pruning_graph
    src = g.edge_index[0]
    first, second = g.triple_index
    assert torch.equal(src[first], src[second])
    assert int((first == second).sum()) == 0  # a bond never pairs with itself


def test_triple_count_matches_the_combinatorial_expectation(pruning_graph):
    """T = sum over atoms of n*(n-1), counting only bonds inside threebody_cutoff."""
    g = pruning_graph
    kept = g.bond_dist <= THREEBODY_CUTOFF
    counts = torch.bincount(g.edge_index[0][kept], minlength=int(g.num_nodes))
    assert int(g.triple_index.size(1)) == int((counts * (counts - 1)).sum())


def test_prune_edges_by_features_returns_ascending_parent_ids(pruning_graph):
    g = pruning_graph
    edge_index, edge_ids = prune_edges_by_features(
        g, feat_name="bond_dist", condition=lambda x: x > THREEBODY_CUTOFF
    )
    assert bool(torch.all(edge_ids[1:] > edge_ids[:-1]))
    assert torch.equal(edge_index, g.edge_index[:, edge_ids])
    assert bool(torch.all(g.bond_dist[edge_ids] <= THREEBODY_CUTOFF))


def test_prune_edges_by_features_rejects_unknown_field(pruning_graph):
    with pytest.raises(ValueError, match="not an edge feature"):
        prune_edges_by_features(pruning_graph, feat_name="not_a_field", condition=lambda x: x > 1)


def test_triples_from_bonds_orders_partners_ascending_within_each_first_bond():
    """Pin the emission order down explicitly, since the three-body scatter depends on the
    first row being sorted and matgl's DGL implementation emits exactly this order."""
    # three bonds all leaving atom 0, plus one leaving atom 1 (which forms no triple)
    src = torch.tensor([0, 0, 0, 1])
    edge_ids = torch.tensor([0, 1, 2, 3])
    triple_index, n_triple_ij = _triples_from_bonds(src, edge_ids, num_nodes=2, num_bonds=4)
    assert triple_index[0].tolist() == [0, 0, 1, 1, 2, 2]
    assert triple_index[1].tolist() == [1, 2, 0, 2, 0, 1]
    assert n_triple_ij.tolist() == [2, 2, 2, 0]


def test_triples_from_bonds_handles_bond_ids_that_are_not_contiguous():
    """After pruning, the surviving bonds are scattered through the parent numbering; the
    emitted ids must be parent ids, not positions in the kept subset."""
    src = torch.tensor([0, 0, 0])
    edge_ids = torch.tensor([2, 5, 9])  # parent-bond ids
    triple_index, n_triple_ij = _triples_from_bonds(src, edge_ids, num_nodes=1, num_bonds=12)
    assert triple_index[0].tolist() == [2, 2, 5, 5, 9, 9]
    assert triple_index[1].tolist() == [5, 9, 2, 9, 2, 5]
    assert n_triple_ij.tolist() == [0, 0, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0]


def test_triples_from_bonds_does_not_assume_bonds_are_grouped_by_source_atom():
    """The DGL implementation relies on the edge list being sorted by source atom. The PyG
    one must not, and must still emit a first row that is ascending."""
    src = torch.tensor([1, 0, 1, 0])  # interleaved
    edge_ids = torch.tensor([0, 1, 2, 3])
    triple_index, n_triple_ij = _triples_from_bonds(src, edge_ids, num_nodes=2, num_bonds=4)
    assert bool(torch.all(triple_index[0][1:] >= triple_index[0][:-1]))
    assert n_triple_ij.tolist() == [1, 1, 1, 1]
    pairs = set(zip(triple_index[0].tolist(), triple_index[1].tolist(), strict=True))
    assert pairs == {(0, 2), (2, 0), (1, 3), (3, 1)}


def test_no_triples_when_every_atom_has_one_bond():
    src = torch.tensor([0, 1])
    edge_ids = torch.tensor([0, 1])
    triple_index, n_triple_ij = _triples_from_bonds(src, edge_ids, num_nodes=2, num_bonds=2)
    assert triple_index.shape == (2, 0)
    assert n_triple_ij.tolist() == [0, 0]


def test_line_graph_with_a_tiny_threebody_cutoff_is_empty(pruning_graph):
    """Every bond pruned: no triples, all-zero n_triple_ij, and the invariants still hold.
    This is the case the old cumsum segment-index helper raised an IndexError on."""
    g = pruning_graph.clone()
    create_line_graph(g, 0.1)
    assert int(g.triple_index.size(1)) == 0
    assert int(g.n_triple_ij.sum()) == 0
    assert int(g.n_triple_ij.numel()) == g.edge_index.size(1)
    assert_lg_invariants(g)


@pytest.mark.parametrize("threebody_cutoff", [1.5, 2.0, 3.0, 4.0, 5.0])
def test_invariants_unbatched(structures, element_types, threebody_cutoff):
    for structure in structures[:8]:
        data, _, _ = build_graph(structure, element_types, threebody_cutoff=threebody_cutoff)
        assert_lg_invariants(data)


def test_invariants_batched(graphs):
    """PyG batching offsets triple_index by the running *bond* count via DIEPData.__inc__.
    Check the offsets explicitly and against a line graph built on the batch directly."""
    datas = [g[0] for g in graphs]
    for batch_size in (2, 3, 5, len(datas)):
        for start in range(0, len(datas) - batch_size + 1, batch_size):
            members = [d.clone() for d in datas[start : start + batch_size]]
            batch = Batch.from_data_list(members)
            assert_lg_invariants(batch)

            offset, first, second, n_triple = 0, [], [], []
            for d in datas[start : start + batch_size]:
                first.append(d.triple_index[0] + offset)
                second.append(d.triple_index[1] + offset)
                n_triple.append(d.n_triple_ij)
                offset += d.edge_index.size(1)
            assert torch.equal(batch.triple_index[0], torch.cat(first))
            assert torch.equal(batch.triple_index[1], torch.cat(second))
            assert torch.equal(batch.n_triple_ij, torch.cat(n_triple))

            direct = Batch.from_data_list([d.clone() for d in datas[start : start + batch_size]])
            direct.triple_index = None
            direct.n_triple_ij = None
            create_line_graph(direct, THREEBODY_CUTOFF)
            assert torch.equal(direct.triple_index, batch.triple_index)
            assert torch.equal(direct.n_triple_ij, batch.n_triple_ij)


def test_assert_lg_invariants_catches_a_pruned_space_line_graph(pruning_graph):
    """Simulate the defect: renumber the triple index into pruned-bond space."""
    g = pruning_graph.clone()
    _, edge_ids = prune_edges_by_features(g, feat_name="bond_dist", condition=lambda x: x > THREEBODY_CUTOFF)
    inverse = torch.zeros(g.edge_index.size(1), dtype=torch.long)
    inverse[edge_ids] = torch.arange(edge_ids.numel())
    g.triple_index = inverse[g.triple_index]  # now pruned-bond ids
    with pytest.raises(AssertionError, match="bincount"):
        assert_lg_invariants(g)


def test_assert_lg_invariants_catches_wrong_length_n_triple_ij(pruning_graph):
    g = pruning_graph.clone()
    g.n_triple_ij = g.n_triple_ij[:-1]
    with pytest.raises(AssertionError, match="parent-bond index space"):
        assert_lg_invariants(g)


def test_assert_lg_invariants_catches_unsorted_first_row(pruning_graph):
    g = pruning_graph.clone()
    perm = torch.randperm(g.triple_index.size(1))
    g.triple_index = g.triple_index[:, perm]
    with pytest.raises(AssertionError, match="sorted ascending"):
        assert_lg_invariants(g)


def test_ensure_line_graph_compatibility_builds_when_absent(prototypes, element_types):
    data, _, _ = build_graph(prototypes[0], element_types, line_graph=False)
    assert getattr(data, "triple_index", None) is None
    ensure_line_graph_compatibility(data, THREEBODY_CUTOFF)
    assert_lg_invariants(data)


def test_ensure_line_graph_compatibility_rejects_a_stale_cache(pruning_graph):
    g = pruning_graph.clone()
    g.n_triple_ij = g.n_triple_ij[:-2]
    with pytest.raises(RuntimeError, match="not compatible"):
        ensure_line_graph_compatibility(g, THREEBODY_CUTOFF)


def test_threebody_cutoff_equal_to_cutoff_prunes_nothing(pruning_graph):
    """The precondition for the index-space defect is threebody_cutoff < cutoff."""
    g = pruning_graph.clone()
    _, edge_ids = prune_edges_by_features(g, feat_name="bond_dist", condition=lambda x: x > CUTOFF)
    assert torch.equal(edge_ids, torch.arange(g.edge_index.size(1)))
    create_line_graph(g, CUTOFF)
    assert_lg_invariants(g)


def test_compute_theta_angles_are_in_range(pruning_graph):
    angles = compute_theta(pruning_graph)
    assert angles["theta"].shape[0] == pruning_graph.triple_index.size(1)
    assert bool((angles["theta"] >= 0).all() and (angles["theta"] <= np.pi).all())
    cosines = compute_theta(pruning_graph, cosine=True)["cos_theta"]
    assert torch.allclose(torch.cos(angles["theta"]), cosines, atol=1e-5)


def test_diepdata_inc_offsets_triple_index_by_bond_count():
    """The one line of batching logic the whole port hangs on."""
    d = DIEPData(edge_index=torch.tensor([[0, 1, 1], [1, 0, 0]]), num_nodes=2)
    assert d.__inc__("triple_index", d.triple_index if hasattr(d, "triple_index") else None) == 3
    assert d.__inc__("edge_index", None) == 2
