"""Regression tests for two batched-path defects fixed on 2026-09-28.

Both were silent on the happy path -- a periodic structure with bonds throughout and no
state features -- which is why they survived until a review exercised the batched paths
directly. Neither is covered by the numerical checks (forces vs finite differences, Hessian
symmetry): those all run on a single structure.

1. ``EmbeddingBlock`` unconditionally unsqueezed the continuous state vector, so a batch's
   ``(B, F)`` became ``(1, B, F)`` and the conv block's ``state_feat[data.batch[src]]``
   indexed past the end. ``include_state=True`` with ``dim_state_feats`` was unusable for
   any batch larger than one structure.

2. The ``field="edge_feat"`` readouts sized their output from ``int(edge_batch.max()) + 1``.
   A structure with no bonds contributes nothing to the edge batch vector, so a batch whose
   *last* structure was edgeless produced one row too few -- a silent misalignment of
   predictions against labels rather than an error.
"""

from __future__ import annotations

import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch import nn
from torch_geometric.data import Batch

from diep_pyg.graph.compute import compute_pair_vector_and_distance, create_line_graph
from diep_pyg.graph.converters import Structure2Graph
from diep_pyg.layers._embedding import EmbeddingBlock
from diep_pyg.models import DIEP

ELEMENTS = ("Si",)
CUTOFF = 5.0
THREEBODY_CUTOFF = 4.0

# Diamond Si primitive cell. A conventional cubic cell with only two atoms is *not* diamond
# and yields a degenerate neighbour list, so build the FCC cell explicitly.
SI = Structure(
    Lattice([[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]]),
    ["Si"] * 2,
    [[0, 0, 0], [0.25, 0.25, 0.25]],
)
# A lone atom in a large box has zero bonds inside the cutoff: the case that exposed (2).
LONE = Structure(Lattice.cubic(30.0), ["Si"], [[0, 0, 0]])


def _graph(structure: Structure):
    """Build a graph with bond geometry and the three-body line graph attached."""
    converter = Structure2Graph(element_types=ELEMENTS, cutoff=CUTOFF)
    g, lattice, _ = converter.get_graph(structure)
    g.pbc_offshift = g.pbc_offset @ lattice[0]
    g.pos = g.frac_coords @ lattice[0]
    g.bond_vec, g.bond_dist = compute_pair_vector_and_distance(g)
    create_line_graph(g, THREEBODY_CUTOFF)
    return g


def _model(**kwargs) -> DIEP:
    base = {
        "element_types": ELEMENTS,
        "integral_mode": "sum",
        "cutoff": CUTOFF,
        "threebody_cutoff": THREEBODY_CUTOFF,
        "nblocks": 1,
        "units": 8,
        "dim_node_embedding": 8,
        "dim_edge_embedding": 8,
        "is_intensive": True,
    }
    return DIEP(**{**base, **kwargs})


def test_lone_atom_really_has_no_bonds():
    """Guard the fixture itself: without this the edge-feat tests below prove nothing."""
    assert _graph(LONE).num_edges == 0


def test_continuous_state_keeps_one_row_per_structure():
    """A ``(B, F)`` state batch must embed to ``(B, D)``, not ``(1, B, D)``."""
    block = EmbeddingBlock(
        degree_rbf=4,
        activation=nn.SiLU(),
        dim_node_embedding=8,
        dim_edge_embedding=8,
        dim_state_feats=4,
        include_state=True,
        ntypes_node=len(ELEMENTS),
    )
    state_attr = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    _, _, state_feat = block(torch.tensor([0, 0]), torch.rand(6, 4), state_attr)
    assert state_feat.shape == (2, 4)


def test_continuous_state_single_structure_still_gains_batch_axis():
    """A bare ``(F,)`` vector still needs the leading structure axis added."""
    block = EmbeddingBlock(
        degree_rbf=4,
        activation=nn.SiLU(),
        dim_node_embedding=8,
        dim_edge_embedding=8,
        dim_state_feats=4,
        include_state=True,
        ntypes_node=len(ELEMENTS),
    )
    _, _, state_feat = block(torch.tensor([0, 0]), torch.rand(6, 4), torch.tensor([0.1, 0.2]))
    assert state_feat.shape == (1, 4)


def test_batched_forward_with_continuous_state():
    """The end-to-end symptom: this raised IndexError from the conv block."""
    model = _model(include_state=True, dim_state_feats=4)
    batch = Batch.from_data_list([_graph(SI), _graph(SI)])
    out = model(batch, state_attr=torch.tensor([[0.1, 0.2], [0.3, 0.4]]))
    assert out.shape == (2,)


def test_batched_forward_with_learned_state_embedding():
    """The ``nn.Embedding`` state branch was never broken; keep it that way."""
    model = _model(include_state=True, ntypes_state=3, dim_state_embedding=4)
    batch = Batch.from_data_list([_graph(SI), _graph(SI)])
    out = model(batch, state_attr=torch.tensor([1, 2]))
    assert out.shape == (2,)


@pytest.mark.parametrize("readout_type", ["weighted_atom", "reduce_atom", "set2set"])
@pytest.mark.parametrize("field", ["node_feat", "edge_feat"])
def test_readout_row_per_structure_with_edgeless_tail(readout_type, field):
    """An edgeless *last* structure must not shorten the readout.

    The tail position matters: an edgeless structure in the middle still leaves a larger
    index behind it, so ``max() + 1`` happens to come out right and the defect hides.
    """
    model = _model(readout_type=readout_type, field=field)
    batch = Batch.from_data_list([_graph(SI), _graph(LONE)])
    assert model(batch).shape == (2,)


@pytest.mark.parametrize("readout_type", ["weighted_atom", "reduce_atom", "set2set"])
@pytest.mark.parametrize("field", ["node_feat", "edge_feat"])
def test_readout_row_per_structure_all_edgeless(readout_type, field):
    """Every structure edgeless: the readout still owes one row each."""
    model = _model(readout_type=readout_type, field=field)
    batch = Batch.from_data_list([_graph(LONE), _graph(LONE)])
    assert model(batch).shape == (2,)


def test_edgeless_tail_does_not_disturb_the_other_structure():
    """The bonded structure's prediction must not depend on what follows it in the batch."""
    model = _model(readout_type="reduce_atom", field="edge_feat").eval()
    with torch.no_grad():
        alone = model(Batch.from_data_list([_graph(SI)]))
        with_tail = model(Batch.from_data_list([_graph(SI), _graph(LONE)]))
    # `forward` ends in torch.squeeze, so a one-structure batch comes back 0-dim.
    torch.testing.assert_close(with_tail[0], alone.reshape(-1)[0])
