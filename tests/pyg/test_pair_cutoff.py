"""Regression coverage for pair-cutoff smoothing in the PyG model."""

from __future__ import annotations

import pytest
import torch

import diep
from diep.pyg.apps.pes import Potential
from diep.pyg.graph.converters import Structure2Graph
from diep.pyg.models import DIEP
from pymatgen.core import Lattice, Structure


@pytest.fixture
def double_precision():
    """Keep endpoint and derivative checks above floating-point cancellation."""
    old_dtype, old_float, old_np = torch.get_default_dtype(), diep.float_th, diep.float_np
    diep.set_default_dtype("float", 64)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)
        diep.float_th, diep.float_np = old_float, old_np


def pair_graph(distance, cutoff=5.0):
    """Rebuild the periodic graph; there are two directed edges inside cutoff."""
    structure = Structure(
        Lattice.cubic(25.0),
        ["Li", "O"],
        [[8.0, 8.0, 8.0], [8.0 + 0.6 * distance, 8.0 + 0.8 * distance, 8.0]],
        coords_are_cartesian=True,
    )
    return Structure2Graph(("Li", "O"), cutoff=cutoff).get_graph(structure)[:2]


def potential_for_mode(mode, use_triplets=True):
    torch.manual_seed(42)
    model = DIEP(
        element_types=("Li", "O"),
        is_intensive=False,
        nblocks=2,
        integral_mode=mode,
        use_triplets=use_triplets,
    )
    return Potential(model=model, calc_forces=True, calc_stresses=False).eval()


@pytest.mark.parametrize("mode", ["grid", "sum"])
@pytest.mark.parametrize("use_triplets", [False, True])
def test_pair_features_and_distance_derivatives_vanish_at_cutoff(double_precision, mode, use_triplets):
    """The actual forward's bond features must join the outside zero smoothly."""
    potential = potential_for_mode(mode, use_triplets)
    # Keep the two edges explicitly to inspect the inner endpoint, independently
    # of the neighbor finder's tolerance for equality at cutoff.
    data, lattice = pair_graph(4.9)
    r = torch.tensor(5.0, requires_grad=True)
    direction = torch.tensor([0.6, 0.8, 0.0])
    data.pos = torch.stack([torch.zeros(3), r * direction])
    data.pbc_offshift = data.pbc_offset @ lattice[0]
    potential.model(g=data)
    expected_width = potential.model.diep_integrator.edge_dim
    assert data.rbf.shape == (2, expected_width)
    assert data.rbf.abs().max().item() < 1e-12
    for channel in data.rbf[0]:
        first = torch.autograd.grad(channel, r, create_graph=True, retain_graph=True)[0]
        second = torch.autograd.grad(first, r, retain_graph=True)[0]
        assert torch.isfinite(first) and torch.isfinite(second)
        assert abs(first.item()) < 1e-10
        assert abs(second.item()) < 1e-10


@pytest.mark.parametrize("mode", ["grid", "sum"])
def test_energy_and_force_approach_no_bond_result(double_precision, mode):
    """Deleting a pair should not leave a finite endpoint jump in energy/force."""
    potential = potential_for_mode(mode)
    energies, forces = [], []
    for distance, expected_edges in ((4.9, 2), (4.99, 2), (5.1, 0)):
        data, lattice = pair_graph(distance)
        assert data.num_edges == expected_edges
        energy, force, _, _ = potential(g=data, lat=lattice)
        assert torch.isfinite(energy).all() and torch.isfinite(force).all()
        energies.append(energy.detach())
        forces.append(force.detach())
    far_energy_gap = abs((energies[0] - energies[2]).item())
    near_energy_gap = abs((energies[1] - energies[2]).item())
    far_force_gap = (forces[0] - forces[2]).abs().max().item()
    near_force_gap = (forces[1] - forces[2]).abs().max().item()
    assert near_energy_gap < far_energy_gap * 0.02 + 1e-12
    assert near_force_gap < far_force_gap * 0.05 + 1e-12
    assert forces[2].abs().max().item() < 1e-12
