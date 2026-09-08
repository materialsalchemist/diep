"""Pure-PyG tests for the DIEP model forward pass: batching consistency and use_edges toggling.

No DGL / stub required -- see tests/pyg/conftest.py.
"""

from __future__ import annotations

import torch
from diep.pyg.graph.compute import assert_lg_invariants, create_line_graph
from diep.pyg.models import DIEP
from torch_geometric.data import Batch


def _model(element_types, **kwargs):
    kwargs.setdefault("nblocks", 2)
    kwargs.setdefault("is_intensive", False)
    kwargs.setdefault("cutoff", 5.0)
    kwargs.setdefault("threebody_cutoff", 4.0)
    model = DIEP(element_types=element_types, **kwargs)
    model.eval()
    return model


def test_forward_runs_with_three_body(graphs, element_types):
    model = _model(element_types)
    data, _, _ = graphs[0]
    with torch.no_grad():
        energy = model(g=data.clone())
    assert torch.isfinite(energy).all()


def test_forward_without_three_body(graphs, element_types):
    model = _model(element_types, use_edges=False)
    data, _, _ = graphs[0]
    data = data.clone()
    del data.triple_index, data.n_triple_ij
    with torch.no_grad():
        energy = model(g=data)
    assert torch.isfinite(energy).all()


def test_batched_forward_matches_per_structure(graphs, element_types):
    model = _model(element_types)
    datas = [d.clone() for d, _, _ in graphs]
    with torch.no_grad():
        single = torch.cat([torch.atleast_1d(model(g=d.clone())) for d in datas])
        batch = Batch.from_data_list(datas)
        assert_lg_invariants(batch)
        batched = torch.atleast_1d(model(g=batch))
    assert torch.allclose(single, batched, atol=1e-4)


def test_batched_forward_matches_per_structure_without_three_body(graphs, element_types):
    model = _model(element_types, use_edges=False)
    datas = []
    for d, _, _ in graphs:
        d = d.clone()
        del d.triple_index, d.n_triple_ij
        datas.append(d)
    with torch.no_grad():
        single = torch.cat([torch.atleast_1d(model(g=d.clone())) for d in datas])
        batched = torch.atleast_1d(model(g=Batch.from_data_list(datas)))
    assert torch.allclose(single, batched, atol=1e-4)


def test_use_edges_toggle_changes_output(graphs, element_types):
    """Sanity: three-body interactions actually affect the energy (the two paths differ)."""
    data, _, _ = graphs[0]

    model_on = _model(element_types, use_edges=True)
    model_off = _model(element_types, use_edges=False)
    model_off.load_state_dict(model_on.state_dict())

    with torch.no_grad():
        e_on = model_on(g=data.clone())
        d_off = data.clone()
        del d_off.triple_index, d_off.n_triple_ij
        e_off = model_off(g=d_off)
    assert not torch.allclose(e_on, e_off)


def test_ensure_line_graph_compatibility_used_when_missing(graphs, element_types):
    """When use_edges is on but the line graph wasn't precomputed, the model builds it."""
    model = _model(element_types)
    data, _, _ = graphs[0]
    data = data.clone()
    del data.triple_index, data.n_triple_ij
    with torch.no_grad():
        energy = model(g=data)
    assert torch.isfinite(energy).all()
    assert hasattr(data, "triple_index")


def test_line_graph_reused_when_already_present(pruning_graph, element_types):
    """When a line graph is already attached, the model does not rebuild it (same object)."""
    data = pruning_graph.clone()
    create_line_graph(data, 4.0)
    triple_index_before = data.triple_index.clone()
    model = _model(element_types)
    with torch.no_grad():
        model(g=data)
    assert torch.equal(data.triple_index, triple_index_before)
