"""Regression tests pinning ``triple_index`` to *parent-bond* index space under batching.

This is the invariant the package exists to hold: ``DIEPData.__inc__`` must offset
``triple_index`` by the running **bond** count, because the line graph's nodes are bonds.
Offsetting by the node count instead is the M3GNet three-body index defect, which produces
no exception, no NaN and a perfectly healthy-looking loss curve -- it just trains on the
wrong triples.

The trap these tests exist to avoid
-----------------------------------
A batching check built from structures whose node count happens to equal their edge count
**cannot distinguish the two offsets at all**: both produce the same number, so the test
passes whether the code is right or wrong. The pre-existing fixtures in
``test_batching_regressions.py`` are exactly that shape (diamond Si is 2 atoms; the lone
atom is 1 atom / 0 edges), so they pin the readout and state defects they were written for
but say nothing about node-vs-bond offsets.

Every fixture below is therefore deliberately **asymmetric**: ``SPARSE`` has 5 nodes and
2 edges, so an offset of 5 (nodes) and an offset of 2 (bonds) are different numbers and the
test can actually fail. ``test_fixtures_are_asymmetric`` guards that property, because if a
later edit makes the counts coincide these tests quietly stop proving anything.
"""

from __future__ import annotations

import random

import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Batch

from diep_pyg.graph.compute import (
    DIEPData,
    assert_lg_invariants,
    compute_pair_vector_and_distance,
    create_line_graph,
)
from diep_pyg.graph.converters import Structure2Graph
from diep_pyg.models import DIEP

ELEMENTS = ("Si", "O")
CUTOFF = 4.0
THREEBODY_CUTOFF = 3.0


def _manual(num_nodes: int, pos: torch.Tensor, edge_index: torch.Tensor) -> DIEPData:
    """A hand-built graph with an exact, chosen (node count, edge count) pair.

    Built by hand rather than from a Structure because the whole point is to control the two
    counts independently; a real neighbour list ties them together.
    """
    data = DIEPData(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_type=torch.zeros(num_nodes, dtype=torch.int32),
        frac_coords=pos.clone(),
        pos=pos.clone(),
        pbc_offset=torch.zeros(edge_index.size(1), 3),
        pbc_offshift=torch.zeros(edge_index.size(1), 3),
        lattice=torch.eye(3).unsqueeze(0),
    )
    data.bond_vec, data.bond_dist = compute_pair_vector_and_distance(data)
    create_line_graph(data, THREEBODY_CUTOFF)
    return data


# 5 nodes, 2 edges: the asymmetry that makes a node-offset bug visible. Only atoms 0 and 1
# are close enough to bond; the rest sit far away and contribute nodes but no edges.
SPARSE_POS = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [8.0, 0, 0], [16.0, 0, 0], [24.0, 0, 0]])
SPARSE_EDGES = torch.tensor([[0, 1], [1, 0]])

# 3 nodes, 6 edges, and a genuine triple on every bond: the opposite asymmetry.
TRIANGLE_POS = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1.0, 0]])
TRIANGLE_EDGES = torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]])


def _sparse() -> DIEPData:
    return _manual(5, SPARSE_POS, SPARSE_EDGES)


def _triangle() -> DIEPData:
    return _manual(3, TRIANGLE_POS, TRIANGLE_EDGES)


def test_fixtures_are_asymmetric():
    """Guard the fixtures: equal node and edge counts would make every test below vacuous.

    Without this, a later edit that "tidied" SPARSE into a 2-atom cell would leave the suite
    green while silently removing its ability to detect a node-vs-bond offset.
    """
    sparse, triangle = _sparse(), _triangle()
    assert sparse.num_nodes == 5 and sparse.edge_index.size(1) == 2
    assert triangle.num_nodes == 3 and triangle.edge_index.size(1) == 6
    assert sparse.num_nodes != sparse.edge_index.size(1)
    assert triangle.num_nodes != triangle.edge_index.size(1)
    # The leading structure's two counts must differ, or the offset is ambiguous.
    assert sparse.num_nodes != sparse.edge_index.size(1)


def test_triple_index_offsets_by_bond_count_not_node_count():
    """The decisive check: 5 nodes but 2 edges ahead of the triples, so the two differ."""
    sparse, triangle = _sparse(), _triangle()
    standalone = triangle.triple_index.clone()
    assert standalone.numel(), "fixture must contribute triples or this proves nothing"

    batch = Batch.from_data_list([sparse, triangle])
    # The second structure's triples start after the first structure's *bonds*.
    shifted = batch.triple_index[:, -standalone.size(1) :]
    expected_bond_offset = sparse.edge_index.size(1)  # 2
    wrong_node_offset = sparse.num_nodes  # 5

    assert expected_bond_offset != wrong_node_offset, "fixture lost its asymmetry"
    torch.testing.assert_close(shifted, standalone + expected_bond_offset)
    assert not torch.equal(shifted, standalone + wrong_node_offset)


def test_batched_triples_still_share_a_centre_atom():
    """Semantic check: the offset must land on the right bonds, not merely in range.

    An offset that is numerically in-bounds can still point at another structure's bonds.
    Every triple's two bonds sharing a source atom is the property that would break.
    """
    batch = Batch.from_data_list([_sparse(), _triangle()])
    src = batch.edge_index[0]
    first, second = batch.triple_index
    assert first.numel()
    assert torch.equal(src[first], src[second])
    assert bool((first != second).all()), "a triple must use two distinct bonds"


def test_edge_level_attributes_stay_one_entry_per_bond():
    """``n_triple_ij`` / ``threebody_cutoff_used`` are edge-level and must match bond count."""
    sparse, triangle = _sparse(), _triangle()
    batch = Batch.from_data_list([sparse, triangle])
    num_bonds = batch.edge_index.size(1)
    assert num_bonds == sparse.edge_index.size(1) + triangle.edge_index.size(1)
    assert int(batch.n_triple_ij.numel()) == num_bonds
    assert int(batch.threebody_cutoff_used.numel()) == num_bonds
    # and the counts must agree with the triples actually present
    expected = torch.bincount(batch.triple_index[0].long(), minlength=num_bonds)
    torch.testing.assert_close(expected.to(batch.n_triple_ij.dtype), batch.n_triple_ij)


def test_batching_prebuilt_line_graphs_equals_building_from_the_batch():
    """The strongest available check, and the one the index defect fails.

    Batching per-structure line graphs must give exactly what enumerating triples over the
    assembled batch gives. These are two independent routes to the same object: if the
    offset were wrong, only the first would be.
    """
    batch = Batch.from_data_list([_sparse(), _triangle()])

    rebuilt = Batch.from_data_list([_sparse(), _triangle()])
    del rebuilt.triple_index, rebuilt.n_triple_ij, rebuilt.threebody_cutoff_used
    create_line_graph(rebuilt, THREEBODY_CUTOFF)

    as_set = lambda t: {tuple(c) for c in t.t().tolist()}  # noqa: E731
    assert as_set(batch.triple_index) == as_set(rebuilt.triple_index)
    torch.testing.assert_close(batch.n_triple_ij, rebuilt.n_triple_ij)


def test_invariants_hold_on_the_asymmetric_batch():
    """``assert_lg_invariants`` is the in-tree guard; run it on the shape that can fail."""
    assert_lg_invariants(Batch.from_data_list([_sparse(), _triangle()]))


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_invariants_hold_in_either_batch_order(order):
    """Which structure leads decides the offset applied to the other, so test both."""
    parts = [_sparse(), _triangle()]
    assert_lg_invariants(Batch.from_data_list([parts[i] for i in order]))


def test_randomised_batches_with_ungrouped_edge_lists():
    """Randomised sweep over shapes, including edge lists not grouped by source atom.

    ``_triples_from_bonds`` explicitly does not assume the edge list is grouped by centre
    atom, unlike the DGL implementation. Shuffling the pairs exercises that claim across
    many node/edge-count combinations rather than the two hand-picked ones above.
    """
    rng = random.Random(7)
    for _ in range(100):
        parts = []
        for _ in range(rng.randint(2, 4)):
            num_nodes = rng.randint(1, 6)
            pos = torch.randn(num_nodes, 3, generator=torch.Generator().manual_seed(rng.randrange(1 << 30))) * 1.5
            pairs = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
            rng.shuffle(pairs)
            pairs = pairs[: rng.randint(0, len(pairs))]
            edge_index = (
                torch.tensor(pairs).t().reshape(2, -1).long() if pairs else torch.zeros((2, 0), dtype=torch.long)
            )
            parts.append(_manual(num_nodes, pos, edge_index))

        batch = Batch.from_data_list(parts)
        assert_lg_invariants(batch)

        src = batch.edge_index[0]
        first, second = batch.triple_index
        if first.numel():
            assert torch.equal(src[first], src[second])

        rebuilt = Batch.from_data_list(parts)
        del rebuilt.triple_index, rebuilt.n_triple_ij, rebuilt.threebody_cutoff_used
        create_line_graph(rebuilt, THREEBODY_CUTOFF)
        as_set = lambda t: {tuple(c) for c in t.t().tolist()}  # noqa: E731
        assert as_set(batch.triple_index) == as_set(rebuilt.triple_index)
        torch.testing.assert_close(batch.n_triple_ij, rebuilt.n_triple_ij)


# --- end to end -------------------------------------------------------------------------

# Two real structures with different atom counts *and* different bond counts, so the
# batched forward pass exercises the offset on a graph the model actually consumes.
S_TWO = Structure(Lattice.cubic(3.2), ["Si", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
S_THREE = Structure(Lattice.cubic(4.1), ["Si", "O", "O"], [[0, 0, 0], [0.4, 0.4, 0.4], [0.7, 0.2, 0.1]])


def _real_graph(structure: Structure):
    converter = Structure2Graph(element_types=ELEMENTS, cutoff=CUTOFF)
    g, lattice, _ = converter.get_graph(structure)
    g.pos = g.frac_coords @ lattice[0]
    g.pbc_offshift = g.pbc_offset @ lattice[0]
    g.bond_vec, g.bond_dist = compute_pair_vector_and_distance(g)
    create_line_graph(g, THREEBODY_CUTOFF)
    return g


def _pes_model() -> DIEP:
    torch.manual_seed(0)
    return DIEP(
        element_types=ELEMENTS,
        integral_mode="sum",
        cutoff=CUTOFF,
        threebody_cutoff=THREEBODY_CUTOFF,
        nblocks=2,
        units=8,
        dim_node_embedding=8,
        dim_edge_embedding=8,
        is_intensive=False,
    ).eval()


def test_real_structures_have_differing_node_and_edge_counts():
    """Guard the end-to-end fixtures the same way, for the same reason."""
    two, three = _real_graph(S_TWO), _real_graph(S_THREE)
    assert two.num_nodes != three.num_nodes
    assert two.edge_index.size(1) != three.edge_index.size(1)
    assert two.triple_index.size(1) and three.triple_index.size(1)


def test_batched_energies_match_individual_energies():
    """A batched prediction must equal the per-structure ones.

    This is the symptom the index defect would finally show as a *number*: cross-structure
    triples change the energy, and nothing else in the pipeline would complain.
    """
    model = _pes_model()
    with torch.no_grad():
        alone_two = model(g=_real_graph(S_TWO)).reshape(-1)
        alone_three = model(g=_real_graph(S_THREE)).reshape(-1)
        batched = model(g=Batch.from_data_list([_real_graph(S_TWO), _real_graph(S_THREE)]))
    torch.testing.assert_close(batched[0], alone_two[0])
    torch.testing.assert_close(batched[1], alone_three[0])


def test_batched_energies_are_order_invariant():
    """Reversing the batch swaps which structure absorbs the offset; results must not move."""
    model = _pes_model()
    with torch.no_grad():
        alone_two = model(g=_real_graph(S_TWO)).reshape(-1)
        alone_three = model(g=_real_graph(S_THREE)).reshape(-1)
        reversed_batch = model(g=Batch.from_data_list([_real_graph(S_THREE), _real_graph(S_TWO)]))
    torch.testing.assert_close(reversed_batch[0], alone_three[0])
    torch.testing.assert_close(reversed_batch[1], alone_two[0])
