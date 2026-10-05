"""Tests for the bond-anchored triplet frame, ``triplet_frame="bond"`` (added 2026-09-30).

The canonical frame (``_canonicalize_triplets_batch``) decides, for each triplet, which edge
lies on the grid's x-axis and which of its ends sits on the left, using edge lengths with
atomic number as tie-break. That decision flips wherever two lengths cross. When the tied
atoms are different elements, the triangle is redrawn mirrored with its charges swapped, so
the energy steps: a trained fold-0 checkpoint stepped by 1.34 meV on an O-Ir-W triplet whose
O-Ir and O-W legs cross at 2.0 A. The canonical frame also puts the third atom at
y = |perp| >= 0, which gives the energy a cone-shaped kink wherever a triplet passes through
180 degrees. On top of that, it builds its y-axis as ``perp / |perp|``, which in float32
magnifies rounding for nearly straight, off-centre triplets. On real MatPES structures this
gave forces off by up to 65 eV/A.

``triplet_frame="bond"`` draws each line-graph entry in the frame of the bond that receives
its message, the way bonds themselves are drawn, and averages each grid map with its mirror
image in y. Nothing is chosen from the geometry.

Where a test is parametrised over both frames, the canonical case is a control. It shows the
geometry really does hit the defect, so the bond case cannot pass vacuously. The float32 test
has no canonical case, because the canonical precision may legitimately be fixed later. On
its geometry the canonical frame measured a gradient error of 2.7e3 against a gradient scale
of 8, while the bond frame measured 4.8e-6.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Data

from diep_pyg import config
from diep_pyg.apps.pes import Potential
from diep_pyg.graph.converters import Structure2Graph
from diep_pyg.layers._diep import DIEPIntegrator
from diep_pyg.layers._diep_core import _bond_anchored_triplets_batch
from diep_pyg.models import DIEP

# Centre O, first-bond end Ir, second-bond end W: two heavy atoms of different Z, so the
# canonical frame's tie-break has something to flip.
Z_TRIPLET = torch.tensor([8, 77, 74])
ANGLE = math.radians(100.0)
ELEMENTS = ("O", "W", "Ir")
CUTOFF = 5.0
THREEBODY_CUTOFF = 4.0


@pytest.fixture
def float64(monkeypatch):
    monkeypatch.setattr(config, "float_th", torch.float64)


def _triplet_graph(pos: torch.Tensor) -> Data:
    """Two bonds out of centre 0, and both line-graph orderings of the triplet they form."""
    edge_index = torch.tensor([[0, 0], [1, 2]])
    bond_vec = pos[edge_index[1]] - pos[edge_index[0]]
    return Data(
        pos=pos,
        edge_index=edge_index,
        bond_vec=bond_vec,
        bond_dist=torch.sqrt((bond_vec**2).sum(dim=1)),
        triple_index=torch.tensor([[0, 1], [1, 0]]),
        num_nodes=3,
    )


def _readout(integrator: DIEPIntegrator) -> torch.Tensor:
    """A fixed random linear readout of the triplet features, standing in for a model."""
    generator = torch.Generator().manual_seed(0)
    # Drawn in float64 and then cast, so float32 and float64 runs share the same numbers.
    return torch.randn(2, integrator.edge_dim, generator=generator, dtype=torch.float64).to(config.float_th)


def _crossing_positions(leg: float) -> list[list[float]]:
    """O at the origin, Ir 2.0 A along x, W at ``leg`` A and 100 degrees from Ir."""
    return [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [leg * math.cos(ANGLE), leg * math.sin(ANGLE), 0.0]]


@pytest.mark.parametrize(("frame", "smooth"), [("canonical", False), ("bond", True)])
def test_no_feature_jump_where_two_legs_cross(float64, frame, smooth):
    """Walk the O-W leg through the O-Ir length; the canonical ordering flips at the crossing.

    Measured: the canonical frame's largest step is 730x its median step (a jump), and the
    bond frame's is 1.0x (every step the same size).
    """
    integrator = DIEPIntegrator(mode="grid", triplet_frame=frame)
    features = torch.stack(
        [
            integrator(_triplet_graph(torch.tensor(_crossing_positions(leg), dtype=torch.float64)), Z_TRIPLET)[1]
            for leg in torch.linspace(1.999, 2.001, 201, dtype=torch.float64).tolist()
        ]
    )
    steps = (features[1:] - features[:-1]).abs().flatten(1).amax(dim=1)
    ratio = float(steps.max() / steps.median())
    if smooth:
        assert ratio < 2.0
    else:
        assert ratio > 100.0


@pytest.mark.parametrize(("frame", "smooth"), [("canonical", False), ("bond", True)])
def test_energy_is_smooth_through_a_straight_triplet(float64, frame, smooth):
    """Lift W off the O-Ir line by h. A cone gives |E - E0| ~ h; a smooth minimum gives h**2.

    Measured slopes |E - E0| / h at h = 1e-2, 1e-3: canonical 1.386, 1.359 (a kink, the force
    flips as W crosses the line); bond 3.8e-2, 3.8e-3 (smooth, thanks to the mirror fold).
    """
    integrator = DIEPIntegrator(mode="grid", triplet_frame=frame)
    readout = _readout(integrator)

    def energy(h: float) -> float:
        pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.5, h, 0.0]], dtype=torch.float64)
        return float((integrator(_triplet_graph(pos), Z_TRIPLET)[1] * readout).sum())

    e0 = energy(0.0)
    slope_coarse, slope_fine = (abs(energy(h) - e0) / h for h in (1e-2, 1e-3))
    ratio = slope_fine / slope_coarse
    if smooth:
        assert ratio < 0.2
    else:
        assert ratio > 0.8


def test_exactly_straight_triplet_has_no_sideways_force_and_a_finite_hessian(float64):
    """At exactly 180 degrees the cross product is exactly zero, as in any rock-salt cell.

    Nothing sideways can be preferred there, so the sideways gradient must vanish, and the
    softened square root must keep the Hessian finite (a plain norm gives NaN). The canonical
    frame measured a sideways gradient of 3.05 here, picked up from its fallback axis.
    """
    integrator = DIEPIntegrator(mode="grid", triplet_frame="bond")
    pos = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [-2.5, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    energy = (integrator(_triplet_graph(pos), Z_TRIPLET)[1] * _readout(integrator)).sum()
    (grad,) = torch.autograd.grad(energy, pos, create_graph=True)
    hessian = torch.stack(
        [torch.autograd.grad(component, pos, retain_graph=True)[0].flatten() for component in grad.flatten()]
    )

    assert grad[:, 0].abs().max() > 1.0, "fixture has no force at all; the sideways check would be vacuous"
    assert float(grad[:, 1:].abs().max()) == 0.0
    assert torch.isfinite(hessian).all()
    assert torch.allclose(hessian, hessian.T, atol=1e-12)


def test_float32_matches_float64_near_180_degrees(monkeypatch):
    """A nearly straight (179.9998 deg), off-centre triplet 20 A from the origin.

    This is the geometry where the canonical frame's ``perp / |perp|`` magnified float32
    rounding into a gradient error of 2.7e3 (on a scale of 8). The bond frame never divides
    by a vanishing length; measured error 4.8e-6.
    """
    theta = math.radians(179.9998)
    generator = torch.Generator().manual_seed(1)
    u = torch.nn.functional.normalize(torch.randn(3, generator=generator, dtype=torch.float64), dim=0)
    v = torch.nn.functional.normalize(
        torch.linalg.cross(u, torch.randn(3, generator=generator, dtype=torch.float64)), dim=0
    )
    centre = torch.tensor([13.7, 21.3, 8.9], dtype=torch.float64)
    positions = torch.stack([centre, centre + 1.8 * u, centre + 3.9 * (math.cos(theta) * u + math.sin(theta) * v)])
    positions = positions.float().double()  # both precisions start from the same inputs

    results = {}
    for dtype in (torch.float32, torch.float64):
        monkeypatch.setattr(config, "float_th", dtype)
        integrator = DIEPIntegrator(mode="grid", triplet_frame="bond")
        pos = positions.to(dtype).requires_grad_(True)
        features = integrator(_triplet_graph(pos), Z_TRIPLET)[1]
        energy = (features * _readout(integrator)).sum()
        results[dtype] = (features.detach().double(), torch.autograd.grad(energy, pos)[0].double())

    (feat32, grad32), (feat64, grad64) = results[torch.float32], results[torch.float64]
    assert (feat32 - feat64).abs().max() < 1e-4 * feat64.abs().max()
    assert (grad32 - grad64).abs().max() < 1e-4 * grad64.abs().max()


def test_bond_frame_is_rotation_and_translation_invariant(float64):
    """Dot and cross products of the two bond vectors are all the frame uses."""
    from scipy.spatial.transform import Rotation

    generator = torch.Generator().manual_seed(0)
    coords = torch.randn(64, 3, 3, generator=generator, dtype=torch.float64) * 2.0
    numbers = torch.randint(1, 90, (64, 3), generator=generator).to(torch.float64)
    reference, reference_numbers = _bond_anchored_triplets_batch(coords, numbers)

    rng = np.random.default_rng(0)
    for _ in range(8):
        rotation = torch.tensor(Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix())
        moved = coords @ rotation.T + torch.tensor(rng.normal(size=3) * 10.0)
        planar, moved_numbers = _bond_anchored_triplets_batch(moved, numbers)
        assert torch.equal(moved_numbers, reference_numbers)
        assert torch.allclose(planar, reference, atol=1e-10)


def test_unknown_triplet_frame_is_rejected():
    with pytest.raises(ValueError, match="triplet_frame"):
        DIEPIntegrator(triplet_frame="longest_edge")


# --- whole model -----------------------------------------------------------------------------


def _model(frame: str) -> DIEP:
    torch.manual_seed(0)
    return DIEP(
        element_types=ELEMENTS,
        integral_mode="grid",
        cutoff=CUTOFF,
        threebody_cutoff=THREEBODY_CUTOFF,
        nblocks=1,
        units=8,
        dim_node_embedding=8,
        dim_edge_embedding=8,
        is_intensive=False,
        triplet_frame=frame,
    ).to(config.float_th)


def _energy_and_forces(model: DIEP, structure: Structure) -> tuple[float, torch.Tensor]:
    g, lattice, _ = Structure2Graph(element_types=ELEMENTS, cutoff=CUTOFF).get_graph(structure)
    energy, forces, _, _ = Potential(model, calc_forces=True, calc_stresses=False)(g, lattice)
    return float(energy), forces.detach()


def _crossing_structure(leg: float) -> Structure:
    """The O-Ir-W triplet of the integrator tests, alone in a 20 A box."""
    shift = np.array([10.0, 10.0, 10.0])
    return Structure(
        Lattice.cubic(20.0),
        ["O", "Ir", "W"],
        np.asarray(_crossing_positions(leg)) + shift,
        coords_are_cartesian=True,
    )


@pytest.mark.parametrize(("frame", "smooth"), [("canonical", False), ("bond", True)])
def test_model_energy_and_force_are_continuous_where_legs_cross(float64, frame, smooth):
    """End to end: the force from autograd must equal the slope of the energy at the crossing.

    Central differences straddle the crossing at O-W = O-Ir = 2.0 A. A jump in E makes them
    disagree with the autograd force by roughly jump / 2h, which grows as h shrinks. A
    continuous energy agrees to O(h**2). Measured relative errors at h = 1e-3, 1e-4, 1e-5:
    canonical 0.024, 0.28, 2.8 (the 1/h signature of a jump); bond 1.6e-7, 2.1e-9, 9.2e-9.
    """
    model = _model(frame)
    h = 1e-4
    e_plus, _ = _energy_and_forces(model, _crossing_structure(2.0 + h))
    e_minus, _ = _energy_and_forces(model, _crossing_structure(2.0 - h))
    _, forces = _energy_and_forces(model, _crossing_structure(2.0))
    radial = torch.tensor([math.cos(ANGLE), math.sin(ANGLE), 0.0], dtype=torch.float64)
    analytic = -float(forces[2] @ radial)  # dE/d(leg) = -F_W . r_hat
    finite_difference = (e_plus - e_minus) / (2 * h)
    # Relative, because a small random-init model has a small energy scale (slope ~1.5e-5).
    assert abs(analytic) > 1e-8, "fixture has no force along the walk; the check would be vacuous"
    relative_error = abs(finite_difference - analytic) / abs(analytic)
    if smooth:
        assert relative_error < 1e-5
    else:
        assert relative_error > 0.05


def test_model_is_invariant_to_atom_relabelling(float64):
    """Anchoring on the first bond must not smuggle enumeration order back in."""
    rng = np.random.default_rng(3)
    species = ["O", "W", "Ir", "O", "O"]
    lattice = Lattice.cubic(4.6)
    while True:
        structure = Structure(lattice, species, rng.random((len(species), 3)))
        distances = structure.distance_matrix[np.triu_indices(len(species), k=1)]
        if distances.min() > 1.6:
            break

    model = _model("bond")
    energy, forces = _energy_and_forces(model, structure)
    for permutation in (np.arange(len(species))[::-1], rng.permutation(len(species))):
        relabelled = Structure(lattice, [species[i] for i in permutation], structure.frac_coords[permutation])
        energy_p, forces_p = _energy_and_forces(model, relabelled)
        assert abs(energy_p - energy) < 1e-10 * max(1.0, abs(energy))
        assert torch.allclose(forces_p, forces[torch.as_tensor(permutation.copy())], atol=1e-10)


def test_checkpoints_cannot_load_into_the_other_frame():
    """Both frames have the same parameter shapes, so only the marker can catch a mix-up."""
    canonical, bond = _model("canonical"), _model("bond")
    # Nothing was added to the canonical state dict, so every existing checkpoint still loads.
    assert DIEPIntegrator(triplet_frame="canonical").state_dict() == {}
    assert set(bond.state_dict()) - set(canonical.state_dict()) == {"diep_integrator.bond_anchored_triplets"}
    assert set(canonical.state_dict()) <= set(bond.state_dict())

    with pytest.raises(RuntimeError, match="bond_anchored_triplets"):
        canonical.load_state_dict(bond.state_dict())
    with pytest.raises(RuntimeError, match="bond_anchored_triplets"):
        bond.load_state_dict(canonical.state_dict())


def test_saved_model_reloads_in_the_bond_frame(tmp_path):
    """``DIEP.load`` rebuilds from the saved init args, which must carry the frame."""
    model = _model("bond")
    structure = _crossing_structure(2.0)
    before = model.predict_structure(structure)
    model.save(tmp_path)

    reloaded = DIEP.load(tmp_path)
    assert reloaded.diep_integrator.triplet_frame == "bond"
    assert torch.equal(reloaded.predict_structure(structure), before)
