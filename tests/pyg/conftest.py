"""Fixtures for the PyG backend tests.

Deliberately independent of the top-level ``tests/conftest.py``, which imports DGL: these
tests must run on machines where DGL is not installable.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from diep.pyg.graph.compute import compute_pair_vector_and_distance, create_line_graph
from diep.pyg.graph.converters import Structure2Graph, get_element_list
from pymatgen.core import Lattice, Structure

CUTOFF = 5.0
THREEBODY_CUTOFF = 4.0  # DIEP's shipped configuration: strictly less than cutoff, so bonds get pruned


@pytest.fixture(scope="session")
def prototypes():
    """A handful of structures spanning several coordination environments."""
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
    ch4 = Structure(
        Lattice.cubic(10.0),
        ["C", "H", "H", "H", "H"],
        np.array([[0, 0, 0], [0.109, 0.109, 0.109], [-0.109, -0.109, 0.109],
                  [0.109, -0.109, -0.109], [-0.109, 0.109, -0.109]]) + 0.5,
    )
    return [si, nacl, mo, ch4]


@pytest.fixture(scope="session")
def structures(prototypes):
    """Prototypes plus rattled variants."""
    rng = np.random.default_rng(42)
    out = []
    for proto in prototypes:
        out.append(proto)
        for _ in range(4):
            s = proto.copy()
            s.perturb(float(rng.uniform(0.02, 0.25)))
            out.append(s)
    return out


@pytest.fixture(scope="session")
def element_types(structures):
    return get_element_list(structures)


def build_graph(structure, element_types, cutoff=CUTOFF, threebody_cutoff=THREEBODY_CUTOFF, line_graph=True):
    """Structure -> PyG graph with bond geometry and (optionally) the three-body line graph."""
    converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
    data, lattice, state_attr = converter.get_graph(structure)
    data.pbc_offshift = torch.matmul(data.pbc_offset, lattice[0])
    data.pos = data.frac_coords @ lattice[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(data)
    data.bond_vec = bond_vec
    data.bond_dist = bond_dist
    if line_graph:
        create_line_graph(data, threebody_cutoff)
    return data, lattice, torch.tensor(state_attr)


@pytest.fixture(scope="session")
def graphs(structures, element_types):
    return [build_graph(s, element_types) for s in structures]


@pytest.fixture(scope="session")
def pruning_graph(prototypes, element_types):
    """A graph where threebody_cutoff < cutoff actually prunes bonds."""
    data, _, _ = build_graph(prototypes[0], element_types)
    assert int((data.bond_dist <= THREEBODY_CUTOFF).sum()) < data.edge_index.size(1)
    return data
