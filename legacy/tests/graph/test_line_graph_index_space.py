"""Regression tests for the three-body line-graph index space.

`create_line_graph` builds the line graph over a *pruned* copy of the atom graph, whose
bonds are renumbered 0..n_kept-1. Everything that consumes the line graph downstream --
`graph.edges()`, `g.edata["bond_vec"]`, `polynomial_cutoff(g.edata["bond_dist"], tbc)` and
the three-body scatter width -- is indexed over the *parent* graph's bonds. These tests pin
down that the line graph leaves `create_line_graph` in parent-bond index space, batched and
unbatched, so that the two spaces cannot silently diverge again.

The failure mode being guarded against is silent: no exception, no NaN, no anomalous loss.
"""

from __future__ import annotations

import dgl
import numpy as np
import pytest
import torch
from diep.ext.pymatgen import Structure2Graph, get_element_list
from diep.graph.compute import (
    _compute_3body,
    assert_lg_invariants,
    compute_pair_vector_and_distance,
    create_line_graph,
    ensure_line_graph_compatibility,
    prune_edges_by_features,
)
from pymatgen.core import Lattice, Structure

CUTOFF = 5.0
THREEBODY_CUTOFF = 4.0  # DIEP's shipped configuration: strictly less than cutoff, so bonds get pruned


def _build_graph(structure, cutoff=CUTOFF, element_types=None):
    element_types = element_types or get_element_list([structure])
    converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
    graph, lattice, _ = converter.get_graph(structure)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lattice[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
    graph.edata["bond_vec"] = bond_vec
    graph.edata["bond_dist"] = bond_dist
    return graph


def _prototypes():
    si = Structure(
        Lattice.cubic(5.43),
        ["Si"] * 8,
        [
            [0, 0, 0], [0.25, 0.25, 0.25], [0, 0.5, 0.5], [0.25, 0.75, 0.75],
            [0.5, 0, 0.5], [0.75, 0.25, 0.75], [0.5, 0.5, 0], [0.75, 0.75, 0.25],
        ],
    )
    nacl = Structure(
        Lattice.cubic(5.64),
        ["Na", "Cl"] * 4,
        [
            [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0],
            [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5],
        ],
    )
    mo = Structure(Lattice.cubic(3.15), ["Mo"] * 2, [[0, 0, 0], [0.5, 0.5, 0.5]])
    return [si, nacl, mo]


@pytest.fixture(scope="module")
def pruning_graph():
    """A graph where threebody_cutoff < cutoff actually prunes bonds.

    The conftest `graph_MoS` fixture happens to have every bond inside 4.0 A, so the two
    index spaces coincide there and the defect cannot fire -- which is exactly the
    "accidentally correct" case the fix prompt warns about.
    """
    g = _build_graph(_prototypes()[0])  # Si8, cutoff 5.0
    assert int((g.edata["bond_dist"] <= THREEBODY_CUTOFF).sum()) < g.num_edges()
    return g


def _rattled(n_per_prototype=8, seed=42):
    """A few dozen structures spanning several coordination environments."""
    rng = np.random.default_rng(seed)
    out = []
    for proto in _prototypes():
        out.append(proto)
        for _ in range(n_per_prototype):
            s = proto.copy()
            s.perturb(float(rng.uniform(0.02, 0.25)))
            out.append(s)
    return out


def test_line_graph_is_in_parent_bond_index_space(pruning_graph):
    """The whole fix in one assertion set: node count, ordering and n_triple_ij."""
    g = pruning_graph
    lg = create_line_graph(g, THREEBODY_CUTOFF)

    assert lg.num_nodes() == g.num_edges()
    assert_lg_invariants(g, lg)

    # ndata is the parent edge data verbatim -- node i of the line graph *is* bond i
    for key in ("bond_dist", "bond_vec", "pbc_offset"):
        assert torch.equal(lg.ndata[key], g.edata[key])


def test_line_graph_ids_match_the_edge_ids_translation_table(pruning_graph):
    """The remap must be exactly `edge_ids`, and `edge_ids` must be ascending."""
    g = pruning_graph
    pruned = prune_edges_by_features(g, feat_name="bond_dist", condition=lambda x: x > THREEBODY_CUTOFF)
    edge_ids = pruned.edata["edge_ids"].reshape(-1).long()

    # ascending order is what makes the remap order-preserving for the three-body scatter
    assert bool(torch.all(edge_ids[1:] > edge_ids[:-1]))

    pruned_lg = _compute_3body(pruned)  # pruned-bond index space (pre-fix behaviour)
    fixed_lg = create_line_graph(g, THREEBODY_CUTOFF)  # parent-bond index space

    src_p, dst_p = pruned_lg.edges()
    src_f, dst_f = fixed_lg.edges()
    assert src_p.numel() == src_f.numel()  # same triples, only renumbered
    assert torch.equal(edge_ids[src_p.long()], src_f.long())
    assert torch.equal(edge_ids[dst_p.long()], dst_f.long())


def test_pruning_actually_happens_for_the_shipped_cutoffs(pruning_graph):
    """Guard the guard: if nothing is pruned the defect cannot fire and the tests below
    would pass vacuously."""
    g = pruning_graph
    n_kept = int((g.edata["bond_dist"] <= THREEBODY_CUTOFF).sum())
    assert n_kept < g.num_edges()
    # and the pre-fix line graph really is in a different index space
    pruned = prune_edges_by_features(g, feat_name="bond_dist", condition=lambda x: x > THREEBODY_CUTOFF)
    assert _compute_3body(pruned).num_nodes() != g.num_edges()


def test_no_triple_references_a_bond_beyond_the_threebody_cutoff(pruning_graph):
    """The defect's loudest symptom: triples referencing bonds where the three-body
    polynomial envelope is an exact zero, which silently deletes them."""
    g = pruning_graph
    lg = create_line_graph(g, THREEBODY_CUTOFF)
    src, dst = lg.edges()
    bond_dist = g.edata["bond_dist"]
    assert int((bond_dist[src.long()] > THREEBODY_CUTOFF).sum()) == 0
    assert int((bond_dist[dst.long()] > THREEBODY_CUTOFF).sum()) == 0


@pytest.mark.parametrize("threebody_cutoff", [2.0, 3.0, 4.0, 5.0])
def test_invariants_unbatched(threebody_cutoff):
    for structure in _rattled(n_per_prototype=3):
        g = _build_graph(structure)
        lg = create_line_graph(g, threebody_cutoff)
        assert_lg_invariants(g, lg)


def test_invariants_batched():
    """dgl.batch offsets line-graph node ids by each line graph's node count. Once that
    count is the parent bond count, the batched line graph is still bond-indexed against
    the batched atom graph -- and matches a manual per-structure construction."""
    structures = _rattled(n_per_prototype=4)
    element_types = get_element_list(structures)
    graphs, line_graphs = [], []
    for structure in structures:
        g = _build_graph(structure, element_types=element_types)
        graphs.append(g)
        line_graphs.append(create_line_graph(g, THREEBODY_CUTOFF))

    for batch_size in (2, 5, len(graphs)):
        for start in range(0, len(graphs) - batch_size + 1, batch_size):
            gs = graphs[start : start + batch_size]
            lgs = line_graphs[start : start + batch_size]
            bg, blg = dgl.batch(gs), dgl.batch(lgs)

            assert_lg_invariants(bg, blg)

            # batching must equal building the line graph on the batched graph directly
            direct = create_line_graph(bg, THREEBODY_CUTOFF)
            assert direct.num_nodes() == blg.num_nodes()
            assert torch.equal(direct.edges()[0].long(), blg.edges()[0].long())
            assert torch.equal(direct.edges()[1].long(), blg.edges()[1].long())
            assert torch.equal(direct.ndata["n_triple_ij"], blg.ndata["n_triple_ij"])


def test_ensure_line_graph_compatibility_is_the_identity():
    """`data.py` pops bond_vec/bond_dist/pbc_offset off cached line graphs; the model
    restores them via ensure_line_graph_compatibility. In parent-bond space that restore
    is the identity, with no cutoff comparison and no tolerance involved."""
    structures = _rattled(n_per_prototype=2)
    element_types = get_element_list(structures)
    graphs = [_build_graph(s, element_types=element_types) for s in structures]
    lgs = [create_line_graph(g, THREEBODY_CUTOFF) for g in graphs]
    for lg in lgs:
        for name in ("bond_vec", "bond_dist", "pbc_offset"):
            lg.ndata.pop(name)

    for g, lg in zip(graphs, lgs, strict=True):
        restored = ensure_line_graph_compatibility(g, lg, THREEBODY_CUTOFF)
        assert_lg_invariants(g, restored)
        for name in ("bond_vec", "bond_dist", "pbc_offset"):
            assert torch.equal(restored.ndata[name], g.edata[name])

    bg, blg = dgl.batch(graphs), dgl.batch(lgs)
    restored = ensure_line_graph_compatibility(bg, blg, THREEBODY_CUTOFF)
    assert_lg_invariants(bg, restored)
    for name in ("bond_vec", "bond_dist", "pbc_offset"):
        assert torch.equal(restored.ndata[name], bg.edata[name])


def test_ensure_line_graph_compatibility_rejects_pruned_space_line_graph(pruning_graph):
    """A line graph cached before the fix is in pruned-bond space and must fail loudly
    rather than being silently reconciled into the wrong index space."""
    g = pruning_graph
    pruned = prune_edges_by_features(g, feat_name="bond_dist", condition=lambda x: x > THREEBODY_CUTOFF)
    stale = _compute_3body(pruned)
    assert stale.num_nodes() != g.num_edges()
    with pytest.raises(RuntimeError, match="one line-graph node per bond"):
        ensure_line_graph_compatibility(g, stale, THREEBODY_CUTOFF)


def test_assert_lg_invariants_catches_a_pruned_space_line_graph(pruning_graph):
    """The helper must actually fire on the pre-fix line graph."""
    g = pruning_graph
    pruned = prune_edges_by_features(g, feat_name="bond_dist", condition=lambda x: x > THREEBODY_CUTOFF)
    stale = _compute_3body(pruned)
    with pytest.raises(AssertionError, match="parent-bond index space"):
        assert_lg_invariants(g, stale)


def test_assert_lg_invariants_catches_bad_n_triple_ij(pruning_graph):
    g = pruning_graph
    lg = create_line_graph(g, THREEBODY_CUTOFF)
    lg.ndata["n_triple_ij"] = torch.zeros_like(lg.ndata["n_triple_ij"])
    with pytest.raises(AssertionError, match="bincount"):
        assert_lg_invariants(g, lg)


def test_threebody_cutoff_equal_to_cutoff_is_unchanged(pruning_graph):
    """The precondition for the defect is threebody_cutoff < cutoff. With them equal
    nothing is pruned, edge_ids == arange, and the remap is a no-op."""
    g = pruning_graph
    lg = create_line_graph(g, CUTOFF)
    pruned = prune_edges_by_features(g, feat_name="bond_dist", condition=lambda x: x > CUTOFF)
    assert pruned.num_edges() == g.num_edges()
    assert torch.equal(pruned.edata["edge_ids"].reshape(-1).long(), torch.arange(g.num_edges()))
    assert_lg_invariants(g, lg)
