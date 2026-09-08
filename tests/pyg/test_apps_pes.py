"""Pure-PyG tests for the Potential app: energies, forces, stresses, chunking, batching.

No DGL / stub required -- see tests/pyg/conftest.py.
"""

from __future__ import annotations

import torch
from diep.pyg.apps.pes import Potential
from diep.pyg.models import DIEP
from torch_geometric.data import Batch


def _model(element_types):
    model = DIEP(
        element_types=element_types, nblocks=2, is_intensive=False, cutoff=5.0, threebody_cutoff=4.0
    )
    model.eval()
    return model


def test_energy_forces_stresses_shapes(graphs, element_types):
    model = _model(element_types)
    pot = Potential(model=model, calc_forces=True, calc_stresses=True)
    data, lat, _ = graphs[0]
    energy, forces, stresses, _ = pot(g=data.clone(), lat=lat)
    assert torch.isfinite(energy).all()
    assert forces.shape == (data.num_nodes, 3)
    assert torch.isfinite(forces).all()
    assert stresses.shape == (3, 3)
    assert torch.isfinite(stresses).all()


def test_forces_sum_to_zero(graphs, element_types):
    """Newton's third law: net force on an isolated periodic structure must vanish."""
    model = _model(element_types)
    pot = Potential(model=model, calc_forces=True, calc_stresses=False)
    data, lat, _ = graphs[0]
    _, forces, _, _ = pot(g=data.clone(), lat=lat)
    assert forces.sum(0).abs().max() < 1e-4


def test_forces_match_numerical_gradient(graphs, element_types):
    model = _model(element_types)
    pot = Potential(model=model, calc_forces=True, calc_stresses=False)
    data, lat, _ = graphs[0]
    _, forces, _, _ = pot(g=data.clone(), lat=lat)

    lat_inv = torch.linalg.inv(lat[0])
    eps = 1e-3
    for atom in range(min(3, data.num_nodes)):
        for axis in range(3):
            shift = torch.zeros(3)
            shift[axis] = eps
            plus, minus = data.clone(), data.clone()
            plus.frac_coords = plus.frac_coords.clone()
            plus.frac_coords[atom] += shift @ lat_inv
            minus.frac_coords = minus.frac_coords.clone()
            minus.frac_coords[atom] -= shift @ lat_inv
            e_plus, _, _, _ = pot(g=plus, lat=lat)
            e_minus, _, _, _ = pot(g=minus, lat=lat)
            numerical = -(float(e_plus.detach()) - float(e_minus.detach())) / (2 * eps)
            assert abs(numerical - float(forces[atom, axis].detach())) < 1e-2


def test_use_edges_none_leaves_model_setting(graphs, element_types):
    """The PyG Potential's use_edges=None default must NOT force three-body off (unlike DGL)."""
    model = _model(element_types)
    assert model.use_edges is True
    pot = Potential(model=model, calc_forces=False, calc_stresses=False)
    assert model.use_edges is True
    data, lat, _ = graphs[0]
    pot(g=data.clone(), lat=lat)
    assert model.use_edges is True


def test_use_edges_false_disables_three_body(graphs, element_types):
    model = _model(element_types)
    pot = Potential(model=model, calc_forces=False, calc_stresses=False, use_edges=False)
    assert model.use_edges is False
    data, lat, _ = graphs[0]
    energy, _, _, _ = pot(g=data.clone(), lat=lat)
    assert torch.isfinite(energy).all()


def test_batched_potential_matches_per_structure(graphs, element_types):
    model = _model(element_types)
    pot = Potential(model=model, calc_forces=True, calc_stresses=True)

    datas = [d.clone() for d, _, _ in graphs[:4]]
    lats = torch.cat([lat for _, lat, _ in graphs[:4]], dim=0)

    energies_single, forces_single = [], []
    for d, lat in zip(datas, lats):
        e, f, _, _ = pot(g=d.clone(), lat=lat.unsqueeze(0))
        energies_single.append(torch.atleast_1d(e))
        forces_single.append(f)
    energies_single = torch.cat(energies_single)
    forces_single = torch.cat(forces_single, dim=0)

    batch = Batch.from_data_list([d.clone() for d in datas])
    e_batch, f_batch, _, _ = pot(g=batch, lat=lats)

    assert torch.allclose(energies_single, torch.atleast_1d(e_batch), atol=1e-4)
    assert torch.allclose(forces_single, f_batch, atol=1e-4)


def test_chunked_forward_runs_and_is_finite(graphs, element_types):
    """Smoke-test the chunked evaluation path.

    Chunking assumes the cell is large relative to the padding (cutoff * nblocks), which
    doesn't hold for these small test cells, so chunked and unchunked energies are not
    expected to match here -- that would need a much larger structure. This only checks the
    chunked code path itself: it runs, and returns finite, correctly-shaped output.
    """
    model = _model(element_types)
    pot_chunked = Potential(
        model=model,
        calc_forces=True,
        calc_stresses=False,
        num_chunks=2,
        apply_chunking=True,
        write_chunk_logs=False,
    )
    data, lat, _ = graphs[0]

    e_chunked, f_chunked, _, _ = pot_chunked(g=data.clone(), lat=lat)
    assert torch.isfinite(torch.atleast_1d(e_chunked)).all()
    assert f_chunked.shape == (data.num_nodes, 3)
    assert torch.isfinite(f_chunked).all()
