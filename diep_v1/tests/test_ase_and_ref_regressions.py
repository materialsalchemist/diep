"""Regression tests for the remaining review defects, fixed 2026-09-29 (fourth pass).

The ASE layer and two latent defects elsewhere. None of these touches training; most are on
the relaxation/MD path, which the numerical checks never exercise.

* ``Atoms2Graph`` branched on ``atoms.pbc.all()``, sending a slab down the isolated-molecule
  path: identity lattice, no image offsets, 89% of neighbours lost, stress divided by 1.
* ``ensemble="npt"`` / ``"npt_berendsen"`` passed ``compressibility_au=None`` into a
  multiplication and always raised ``TypeError``.
* ``pfactor`` was ``ptime**2 * units.fs`` where ASE defines it as ``ptime**2 * B`` -- the time
  factor unsquared and the bulk modulus missing, leaving the barostat ~17x too stiff.
* ``upper_triangular_cell`` rotated positions but not momenta.
* ``set_atoms`` left the new Atoms at rest and the step counter stale.
* ``TrajectoryObserver`` appended stress/cell conditionally while indexing all five lists by
  frame, so ``obs[0]`` and ``as_pandas()`` raised whenever either was absent; and the explicit
  ``obs()`` after ``run()`` duplicated the final frame.
* ``AtomRef`` with a 2D ``property_offset`` indexed the state axis, returning (B, B).
* ``allow_missing_labels=True`` crashed whenever stress was disabled.

Each test was confirmed to fail with its own fix reverted.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase import units
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torch_geometric.data import Batch, Data

from diep_pyg.apps.pes import Potential
from diep_pyg.ext.ase import Atoms2Graph, MolecularDynamics, Relaxer
from diep_pyg.layers._atom_ref import AtomRef
from diep_pyg.models._diep import DIEP
from diep_pyg.utils.training import PotentialLightningModule


@pytest.fixture(scope="module")
def potential():
    """A small randomly-initialised potential; these tests check plumbing, not accuracy."""
    torch.manual_seed(0)
    model = DIEP(element_types=("Si",), nblocks=1, cutoff=4.0, threebody_cutoff=3.0)
    pot = Potential(model, calc_forces=True, calc_stresses=True)
    pot.eval()
    return pot


# --------------------------------------------------------------------------- mixed PBC


def test_mixed_pbc_keeps_the_real_cell_and_its_periodic_neighbours():
    """A slab must stay periodic in x/y, not collapse to an isolated cluster.

    The old ``pbc.all()`` branch gave a 3-layer Si slab 18 edges out of 168 and an identity
    lattice, so the stress was divided by a unit volume instead of 120.08 A^3.
    """
    converter = Atoms2Graph(("Si",), 5.0)
    atoms = bulk("Si", "diamond", 5.43) * (1, 1, 3)

    full = atoms.copy()
    full.pbc = [True, True, True]
    slab = atoms.copy()
    slab.pbc = [True, True, False]
    isolated = atoms.copy()
    isolated.pbc = [False, False, False]

    counts = {}
    for name, case in (("full", full), ("slab", slab), ("isolated", isolated)):
        graph, lattice, _ = converter.get_graph(case)
        counts[name] = graph.edge_index.size(1)
        if name != "isolated":
            volume = abs(float(np.linalg.det(np.asarray(lattice[0]))))
            assert volume == pytest.approx(case.get_volume(), rel=1e-9), (
                f"{name}: graph lattice volume {volume} != real {case.get_volume()}"
            )

    # Mixed periodicity sits strictly between the two extremes: more than isolated (it keeps
    # the in-plane images) and fewer than fully periodic (no images along z).
    assert counts["isolated"] < counts["slab"] < counts["full"], counts


def test_singular_cell_with_pbc_raises_instead_of_dividing_by_zero():
    """Periodic flags with a zero-volume cell cannot give a meaningful stress."""
    converter = Atoms2Graph(("Si",), 5.0)
    atoms = Atoms("Si2", positions=[[0, 0, 0], [0, 0, 2.3]], cell=np.zeros((3, 3)), pbc=True)
    with pytest.raises(ValueError, match="singular"):
        converter.get_graph(atoms)


# --------------------------------------------------------------------------- MD parameters


@pytest.mark.parametrize("ensemble", ["npt", "npt_berendsen"])
def test_berendsen_npt_runs_on_its_own_defaults(potential, ensemble):
    """These ensembles multiply by ``compressibility_au``, so None was always a TypeError."""
    atoms = bulk("Si", "diamond", 5.43) * (2, 2, 2)
    md = MolecularDynamics(atoms, potential=potential, ensemble=ensemble, temperature=300, timestep=1.0)
    md.run(2)


def test_pfactor_is_ptime_squared_times_bulk_modulus(potential):
    """ASE defines ``pfactor = ptime**2 * B``; the old code passed ``ptime**2 * units.fs``.

    That gave 552.53 instead of 32.56 -- a factor of 16.97, so the barostat timescale was
    ~309 fs rather than the requested 75 fs.
    """
    atoms = bulk("Si", "diamond", 5.43) * (2, 2, 2)
    md = MolecularDynamics(
        atoms, potential=potential, ensemble="npt_nose_hoover", temperature=300, timestep=1.0,
        ptime=75.0, bulk_modulus=0.6,
    )
    expected = (75.0 * units.fs) ** 2 * 0.6
    assert md.dyn.pfactor_given == pytest.approx(expected, rel=1e-9)
    # The defect's value, guarded explicitly so a revert cannot pass.
    assert md.dyn.pfactor_given != pytest.approx(75.0**2 * units.fs, rel=1e-9)


def test_explicit_pfactor_overrides_ptime_and_bulk_modulus(potential):
    """A caller who passes ``pfactor`` directly must get exactly that value."""
    atoms = bulk("Si", "diamond", 5.43) * (2, 2, 2)
    md = MolecularDynamics(
        atoms, potential=potential, ensemble="npt_nose_hoover", temperature=300, timestep=1.0,
        pfactor=123.456,
    )
    assert md.dyn.pfactor_given == pytest.approx(123.456, rel=1e-12)


def test_upper_triangular_cell_rotates_momenta_with_positions(potential):
    """The cell transform must carry velocities into the new frame.

    ``set_cell(scale_atoms=True)`` moves positions and leaves momenta behind, so velocities end
    up misoriented relative to the lattice. The transform is a rigid rotation, so the check is
    that kinetic energy survives while the momenta themselves change.
    """
    atoms = bulk("Si", "diamond", 5.43)  # FCC primitive: not upper-triangular
    MaxwellBoltzmannDistribution(atoms, temperature_K=300)
    md = MolecularDynamics(atoms, potential=potential, ensemble="nvt", temperature=300, timestep=1.0)

    before = md.atoms.get_momenta().copy()
    kinetic_before = md.atoms.get_kinetic_energy()
    md.upper_triangular_cell()
    after = md.atoms.get_momenta()

    assert not np.allclose(before, after), "momenta were left in the old frame"
    assert md.atoms.get_kinetic_energy() == pytest.approx(kinetic_before, rel=1e-9), (
        "a rigid rotation must preserve kinetic energy"
    )


def test_set_atoms_reseeds_velocities_and_restarts_the_step_counter(potential):
    """A new Atoms is a new run: it needs velocities of its own and a fresh step count."""
    atoms = bulk("Si", "diamond", 5.43) * (2, 2, 2)
    md = MolecularDynamics(atoms, potential=potential, ensemble="nvt", temperature=300, timestep=1.0)
    md.run(2)
    assert md.dyn.nsteps == 2

    replacement = bulk("Si", "diamond", 5.43) * (1, 1, 2)
    assert replacement.get_kinetic_energy() == pytest.approx(0.0), "precondition: arrives at rest"
    md.set_atoms(replacement)

    assert md.dyn.nsteps == 0, "step counter carried over from the previous run"
    assert md.atoms.get_kinetic_energy() > 0.0, "new atoms left at rest; the next run divides by T=0"
    md.run(2)  # used to raise ZeroDivisionError inside the thermostat


def test_set_atoms_keeps_caller_supplied_velocities(potential):
    """Reseeding must not overwrite velocities the caller deliberately set."""
    atoms = bulk("Si", "diamond", 5.43) * (2, 2, 2)
    md = MolecularDynamics(atoms, potential=potential, ensemble="nvt", temperature=300, timestep=1.0)

    replacement = bulk("Si", "diamond", 5.43) * (1, 1, 2)
    MaxwellBoltzmannDistribution(replacement, temperature_K=900)
    supplied = replacement.get_momenta().copy()
    md.set_atoms(replacement)

    assert np.allclose(md.atoms.get_momenta(), supplied), "caller's velocities were overwritten"


# --------------------------------------------------------------------------- trajectory


def test_trajectory_lists_stay_parallel_without_stress(potential):
    """``obs[0]`` and ``as_pandas()`` index all five lists, so all five must track frames."""
    model = DIEP(element_types=("Si",), nblocks=1, cutoff=4.0, threebody_cutoff=3.0)
    no_stress = Potential(model, calc_forces=True, calc_stresses=False)
    no_stress.eval()

    atoms = bulk("Si", "diamond", 5.43)
    atoms.rattle(0.05, seed=1)
    observer = Relaxer(no_stress, relax_cell=False).relax(atoms, fmax=0.05, steps=3)["trajectory"]

    lengths = {
        len(observer.energies),
        len(observer.forces),
        len(observer.stresses),
        len(observer.cells),
        len(observer.atom_positions),
    }
    assert len(lengths) == 1, f"list lengths diverged: {lengths}"
    observer[0]  # used to raise IndexError
    assert observer.as_pandas().shape[0] == len(observer.energies)


def test_trajectory_lists_stay_parallel_without_a_cell(potential):
    """Same for a non-periodic system, where the cell was the missing list."""
    atoms = Atoms("Si2", positions=[[0, 0, 0], [0, 0, 2.3]], pbc=False)
    observer = Relaxer(potential, relax_cell=False).relax(atoms, fmax=0.05, steps=3)["trajectory"]

    lengths = {len(observer.energies), len(observer.cells), len(observer.atom_positions)}
    assert len(lengths) == 1, f"list lengths diverged: {lengths}"
    assert observer.as_pandas().shape[0] == len(observer.energies)


@pytest.mark.parametrize("interval", [1, 2, 3])
def test_final_frame_is_recorded_exactly_once(potential, interval):
    """The attached observer already fires on the last step at interval=1.

    An unconditional ``obs()`` after ``run()`` recorded it twice, double-weighting the final
    configuration in any per-frame statistic. For interval > 1 the call is still needed, so the
    tail must be present either way.
    """
    atoms = bulk("Si", "diamond", 5.43) * (2, 1, 1)
    atoms.rattle(0.2, seed=3)
    # fmax it cannot reach, so the run is cut off by `steps` and exercises the tail logic.
    observer = Relaxer(potential, relax_cell=False).relax(
        atoms, fmax=1e-8, steps=7, interval=interval
    )["trajectory"]

    assert len(observer.energies) > 1
    assert not np.allclose(observer.atom_positions[-1], observer.atom_positions[-2]), (
        "final configuration recorded twice"
    )
    lengths = {len(observer.energies), len(observer.stresses), len(observer.atom_positions)}
    assert len(lengths) == 1, f"list lengths diverged: {lengths}"


# --------------------------------------------------------------------------- AtomRef


def _batch_of(sizes, atomic_number=14):
    graphs = []
    for n in sizes:
        data = Data()
        data.node_type = torch.full((n,), atomic_number, dtype=torch.long)
        data.num_nodes = n
        graphs.append(data)
    return Batch.from_data_list(graphs)


def test_atom_ref_2d_gathers_one_offset_per_structure():
    """A 2D ``property_offset`` must return (B,), not the (B, B) the state-axis index gave."""
    offsets = torch.zeros(3, 95)
    offsets[0, 14], offsets[1, 14], offsets[2, 14] = -5.0, -7.0, -9.0
    atom_ref = AtomRef(property_offset=offsets)

    sizes = (2, 3, 4, 5)
    batch = _batch_of(sizes)
    states = torch.tensor([0, 1, 2, 0])
    result = atom_ref(batch, state_attr=states)

    assert result.shape == (len(sizes),), f"expected one offset per structure, got {tuple(result.shape)}"
    expected = torch.tensor([2 * -5.0, 3 * -7.0, 4 * -9.0, 5 * -5.0])
    assert torch.allclose(result, expected), f"{result.tolist()} != {expected.tolist()}"


def test_atom_ref_2d_without_state_attr_raises():
    """``state_attr=None`` used to *add* an axis, silently giving (1, n_states, B)."""
    offsets = torch.zeros(2, 95)
    offsets[0, 14], offsets[1, 14] = -5.0, -7.0
    atom_ref = AtomRef(property_offset=offsets)
    with pytest.raises(ValueError, match="state_attr is required"):
        atom_ref(_batch_of((2, 3)))


def test_atom_ref_1d_path_is_unchanged():
    """The ordinary 1D table -- the only thing ``fit`` produces -- must behave as before."""
    offsets = torch.zeros(95)
    offsets[14] = -5.0
    atom_ref = AtomRef(property_offset=offsets)
    result = atom_ref(_batch_of((2, 3, 4)))
    assert torch.allclose(result, torch.tensor([-10.0, -15.0, -20.0]))


# --------------------------------------------------------------------------- missing labels


def test_allow_missing_labels_works_with_stress_disabled():
    """A disabled target has a placeholder prediction the label's mask cannot index."""
    module = PotentialLightningModule(
        model=DIEP(element_types=("Si",), nblocks=1), stress_weight=0.0, allow_missing_labels=True
    )
    n_structures, n_atoms = 4, 10
    labels = (torch.randn(n_structures), torch.randn(n_atoms, 3), torch.zeros(n_structures))
    preds = (torch.randn(n_structures), torch.randn(n_atoms, 3), torch.zeros(1))

    results, _ = module.loss_fn(
        loss=torch.nn.MSELoss(), labels=labels, preds=preds,
        num_atoms=torch.full((n_structures,), float(n_atoms) / n_structures),
    )
    assert torch.isfinite(results["Total_Loss"])


def test_allow_missing_labels_still_masks_nan_energies():
    """The masking itself must keep working; the fix only skips unconsumed targets."""
    module = PotentialLightningModule(
        model=DIEP(element_types=("Si",), nblocks=1), stress_weight=0.0, allow_missing_labels=True
    )
    n_structures, n_atoms = 4, 10
    energies = torch.tensor([1.0, float("nan"), 3.0, 4.0])
    labels = (energies, torch.randn(n_atoms, 3), torch.zeros(n_structures))
    preds = (torch.randn(n_structures), torch.randn(n_atoms, 3), torch.zeros(1))

    _, weights = module.loss_fn(
        loss=torch.nn.MSELoss(), labels=labels, preds=preds,
        num_atoms=torch.full((n_structures,), float(n_atoms) / n_structures),
    )
    assert weights["Energy_MAE"] == 3, "the NaN energy label was not masked out"
