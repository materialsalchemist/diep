"""Numerical parity between the DGL and PyG DIEP backends.

Requires DGL -- or, on machines without a DGL wheel (e.g. linux-aarch64), the test stub in
tests/_dgl_stub.py, opted into via DIEP_DGL_STUB=1 (see tests/conftest.py). If neither is
available the whole module is skipped: parity between backends cannot be checked without a
working DGL side to compare against, but that is exactly what tests/pyg/ is for -- it needs
neither backend and is not affected by this skip.

These tests load the pretrained ``diep_pes`` weights and check, structure by structure and
also batched, that:
  * the state dict trained on the DGL model loads into the PyG model with zero missing/
    unexpected keys (the checkpoint-compatibility claim the port depends on);
  * the line graph (edge_index / triple_index / n_triple_ij / bond_dist) is identical between
    backends, i.e. both are in the same parent-bond index space;
  * DIEPIntegrator's bond/triplet features and the model's energy match to float precision;
  * Potential's energies, forces and stresses match under both DGL and PyG batching.
"""

from __future__ import annotations

import json
import os

import pytest
import torch

dgl = pytest.importorskip("dgl", reason="parity tests need DGL or DIEP_DGL_STUB=1 to compare against")

import diep  # noqa: E402
from diep.ext.pymatgen import Structure2Graph as DGLStructure2Graph  # noqa: E402
from diep.graph.compute import compute_pair_vector_and_distance as dgl_bonds  # noqa: E402
from diep.graph.compute import assert_lg_invariants as dgl_assert  # noqa: E402
from diep.graph.compute import create_line_graph as dgl_line_graph  # noqa: E402
from diep.models._diep import DIEP as DGLDIEP  # noqa: E402
from diep.apps.pes import Potential as DGLPotential  # noqa: E402
from diep.pyg.graph.converters import Structure2Graph as PyGStructure2Graph  # noqa: E402
from diep.pyg.graph.converters import get_element_list  # noqa: E402
from diep.pyg.graph.compute import compute_pair_vector_and_distance as pyg_bonds  # noqa: E402
from diep.pyg.graph.compute import assert_lg_invariants as pyg_assert  # noqa: E402
from diep.pyg.graph.compute import create_line_graph as pyg_line_graph  # noqa: E402
from diep.pyg.models import DIEP as PyGDIEP  # noqa: E402
from diep.pyg.apps.pes import Potential as PyGPotential  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PRETRAINED_DIR = os.path.join(REPO_ROOT, "pretrained_models", "diep_pes")
TEST_STRUCTURES_DIR = os.path.join(REPO_ROOT, "test_structures")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(PRETRAINED_DIR), reason=f"pretrained weights not found at {PRETRAINED_DIR}"
)


def _structures():
    out = []
    if os.path.isdir(TEST_STRUCTURES_DIR):
        for fname in sorted(os.listdir(TEST_STRUCTURES_DIR)):
            if fname.endswith(".cif"):
                out.append((fname[:-4], Structure.from_file(os.path.join(TEST_STRUCTURES_DIR, fname))))
    si = Structure(
        Lattice.cubic(5.43),
        ["Si"] * 8,
        [
            [0, 0, 0], [0.25, 0.25, 0.25], [0, 0.5, 0.5], [0.25, 0.75, 0.75],
            [0.5, 0, 0.5], [0.75, 0.25, 0.75], [0.5, 0.5, 0], [0.75, 0.75, 0.25],
        ],
    )
    rattled = si.copy()
    rattled.perturb(0.15)
    out.append(("Si8", si))
    out.append(("Si8-rattled", rattled))
    out.append(("Mo2", Structure(Lattice.cubic(3.15), ["Mo"] * 2, [[0, 0, 0], [0.5, 0.5, 0.5]])))
    out.append((
        "NaCl8",
        Structure(
            Lattice.cubic(5.64),
            ["Na", "Cl"] * 4,
            [
                [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0],
                [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5],
            ],
        ),
    ))
    return out


STRUCTURES = _structures()
CUTOFF, THREEBODY_CUTOFF = 5.0, 4.0


@pytest.fixture(scope="module")
def pretrained_config():
    with open(os.path.join(PRETRAINED_DIR, "model.json")) as f:
        cfg = json.load(f)
    init_args = cfg["kwargs"]["model"]["init_args"]
    state = torch.load(os.path.join(PRETRAINED_DIR, "state.pt"), map_location="cpu", weights_only=False)
    model_state_dict = {k[len("model.") :]: v for k, v in state.items() if k.startswith("model.")}
    return init_args, model_state_dict


@pytest.fixture(scope="module")
def dgl_model(pretrained_config):
    init_args, msd = pretrained_config
    model = DGLDIEP(**init_args)
    result = model.load_state_dict(msd, strict=False)
    assert not result.missing_keys and not result.unexpected_keys, result
    model.eval()
    model.use_edges = model.use_triplets = True
    return model


@pytest.fixture(scope="module")
def pyg_model(pretrained_config):
    init_args, msd = pretrained_config
    model = PyGDIEP(**init_args)
    result = model.load_state_dict(msd, strict=False)
    assert not result.missing_keys and not result.unexpected_keys, result
    model.eval()
    model.use_edges = model.use_triplets = True
    return model


def test_pretrained_dgl_state_dict_loads_into_pyg_model(pretrained_config):
    """The checkpoint compatibility claim the port depends on: no missing/unexpected keys."""
    init_args, msd = pretrained_config
    model = PyGDIEP(**init_args)
    result = model.load_state_dict(msd, strict=False)
    assert result.missing_keys == []
    assert result.unexpected_keys == []


def _build_dgl(structure, element_types):
    graph, lattice, state = DGLStructure2Graph(element_types=element_types, cutoff=CUTOFF).get_graph(structure)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
    graph.ndata["pos"] = graph.ndata["frac_coords"] @ lattice[0]
    bond_vec, bond_dist = dgl_bonds(graph)
    graph.edata["bond_vec"], graph.edata["bond_dist"] = bond_vec, bond_dist
    line_graph = dgl_line_graph(graph, THREEBODY_CUTOFF)
    dgl_assert(graph, line_graph)
    return graph, lattice, state, line_graph


def _build_pyg(structure, element_types):
    data, lattice, state = PyGStructure2Graph(element_types=element_types, cutoff=CUTOFF).get_graph(structure)
    data.pbc_offshift = torch.matmul(data.pbc_offset, lattice[0])
    data.pos = data.frac_coords @ lattice[0]
    bond_vec, bond_dist = pyg_bonds(data)
    data.bond_vec, data.bond_dist = bond_vec, bond_dist
    pyg_line_graph(data, THREEBODY_CUTOFF)
    pyg_assert(data)
    return data, lattice, state


@pytest.mark.parametrize("name,structure", STRUCTURES, ids=[n for n, _ in STRUCTURES])
def test_line_graph_identical_across_backends(name, structure):
    element_types = get_element_list([structure])
    dgl_graph, _, _, dgl_lg = _build_dgl(structure, element_types)
    pyg_data, _, _ = _build_pyg(structure, element_types)

    assert torch.equal(torch.stack(dgl_graph.edges()).long(), pyg_data.edge_index)
    assert torch.equal(torch.stack(dgl_lg.edges()).long(), pyg_data.triple_index)
    assert torch.equal(dgl_lg.ndata["n_triple_ij"], pyg_data.n_triple_ij)
    assert torch.allclose(dgl_graph.edata["bond_dist"], pyg_data.bond_dist, atol=0, rtol=0)


@pytest.mark.parametrize("name,structure", STRUCTURES, ids=[n for n, _ in STRUCTURES])
def test_model_energy_matches_across_backends(name, structure, dgl_model, pyg_model):
    els = dgl_model.element_types  # the pretrained model's element list, not the structure's own

    dgl_graph, _, _, dgl_lg = _build_dgl(structure, els)
    pyg_data, _, _ = _build_pyg(structure, els)

    with torch.no_grad():
        e_dgl = dgl_model(g=dgl_graph, state_attr=None, l_g=dgl_lg)
        e_pyg = pyg_model(g=pyg_data, state_attr=None)

    assert torch.allclose(e_dgl, e_pyg, atol=1e-4), (name, float(e_dgl), float(e_pyg))


def test_potential_energy_forces_stresses_match(dgl_model, pyg_model):
    dgl_pot = DGLPotential(model=dgl_model, calc_forces=True, calc_stresses=True, use_edges=True, num_chunks=0)
    pyg_pot = PyGPotential(model=pyg_model, calc_forces=True, calc_stresses=True, use_edges=True)
    dgl_pot.eval()
    pyg_pot.eval()

    els = dgl_model.element_types
    for name, structure in STRUCTURES:
        dgl_graph, lattice, _ = DGLStructure2Graph(element_types=els, cutoff=CUTOFF).get_graph(structure)
        pyg_data, pyg_lattice, _ = PyGStructure2Graph(element_types=els, cutoff=CUTOFF).get_graph(structure)

        e_dgl, f_dgl, s_dgl, _ = dgl_pot(g=dgl_graph, lat=lattice)
        e_pyg, f_pyg, s_pyg, _ = pyg_pot(g=pyg_data, lat=pyg_lattice)

        assert torch.allclose(e_dgl, e_pyg, atol=1e-4), name
        assert torch.allclose(f_dgl, f_pyg, atol=1e-4), name
        assert torch.allclose(s_dgl, s_pyg, atol=1e-3), name


def test_potential_batched_matches_across_backends(dgl_model, pyg_model):
    dgl_pot = DGLPotential(model=dgl_model, calc_forces=True, calc_stresses=True, use_edges=True, num_chunks=0)
    pyg_pot = PyGPotential(model=pyg_model, calc_forces=True, calc_stresses=True, use_edges=True)
    dgl_pot.eval()
    pyg_pot.eval()

    els = dgl_model.element_types
    dgl_graphs, lattices, pyg_datas = [], [], []
    for _, structure in STRUCTURES:
        g, lat, _ = DGLStructure2Graph(element_types=els, cutoff=CUTOFF).get_graph(structure)
        d, _, _ = PyGStructure2Graph(element_types=els, cutoff=CUTOFF).get_graph(structure)
        dgl_graphs.append(g)
        lattices.append(lat)
        pyg_datas.append(d)

    batched_lattice = torch.cat(lattices, dim=0)
    e_dgl, f_dgl, s_dgl, _ = dgl_pot(g=dgl.batch(dgl_graphs), lat=batched_lattice)
    e_pyg, f_pyg, s_pyg, _ = pyg_pot(g=Batch.from_data_list(pyg_datas), lat=batched_lattice)

    assert torch.allclose(e_dgl, e_pyg, atol=1e-4)
    assert torch.allclose(f_dgl, f_pyg, atol=1e-4)
    assert torch.allclose(s_dgl, s_pyg, atol=1e-3)
