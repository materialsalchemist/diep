"""Checks on the released model and on the pipeline that produced it.

    pytest tests/test_release.py -q

The reference predictions in ``data/reference_predictions.json`` were written by this
release at export time. Regenerate them (only after a deliberate change) with
``python tests/test_release.py --regen``.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "diep_fold1"
REFERENCE = Path(__file__).resolve().parent / "data" / "reference_predictions.json"
sys.path.insert(0, str(ROOT))

from diep_pyg import export as E  # noqa: E402
from diep_pyg.pretrained import load_potential, make_calculator, model_info  # noqa: E402


@pytest.fixture(scope="module")
def pot():
    return load_potential(MODEL)


@pytest.fixture()
def one_thread():
    # Multithreaded CPU backward accumulates in a nondeterministic order; bitwise and
    # tight comparisons need one thread.
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)


def test_model_matches_its_provenance(pot):
    info = model_info(MODEL)
    meta = json.loads((MODEL / "fold1" / "meta.json").read_text())
    assert info["fold"] == meta["fold"] == 1
    assert sum(p.numel() for p in pot.model.parameters()) == info["n_parameters"] == 403181
    assert pot.model.cutoff == meta["cutoff"] == 5.0
    assert pot.model.threebody_cutoff == meta["threebody_cutoff"] == 4.0
    assert tuple(pot.model.element_types) == tuple(meta["element_types"])
    assert pot.model.use_triplets is True
    assert pot.model.diep_integrator.triplet_frame == "canonical"
    refs = np.load(MODEL / "fold1" / "element_refs.npy")
    assert np.allclose(pot.element_refs.property_offset.numpy(), refs, rtol=0, atol=1e-5)
    assert float(pot.data_std) == pytest.approx(meta["force_rms"], abs=1e-6)


def test_model_files_load_without_pickle():
    # Forward-compatible with torch >= 2.6, where weights_only=True is the default.
    torch.load(MODEL / "model.pt", map_location="cpu", weights_only=True)
    torch.load(MODEL / "state.pt", map_location="cpu", weights_only=True)


def test_exported_model_equals_training_checkpoint(pot, one_thread):
    run = json.loads((MODEL / "training" / "metrics.json").read_text())
    ckpt = MODEL / "checkpoint" / "best-epoch=0295.ckpt"
    src, blob, _, _ = E.load_checkpoint_potential(ckpt, MODEL / "fold1", run)
    assert blob["epoch"] == model_info(MODEL)["epoch"] == 295
    E._assert_same_state(src, pot, "test")
    E._assert_same_outputs(src.eval(), pot, E._probe_structures(), "test")  # raises on any bit


def _reference_predictions(pot) -> list[dict]:
    out = []
    for s in E._probe_structures():
        e, f, st = E._predict(pot, s)
        out.append({"formula": s.composition.reduced_formula, "energy": float(e),
                    "forces": f.tolist(), "stress_GPa": st.tolist()})
    return out


def test_reference_predictions(pot, one_thread):
    ref = json.loads(REFERENCE.read_text())
    got = _reference_predictions(pot)
    for r, g in zip(ref, got, strict=True):
        assert g["formula"] == r["formula"]
        # float32 on a different BLAS or CPU can move the last digits, nothing more.
        assert g["energy"] == pytest.approx(r["energy"], rel=1e-5, abs=1e-4)
        np.testing.assert_allclose(g["forces"], r["forces"], rtol=1e-4, atol=1e-3)
        np.testing.assert_allclose(g["stress_GPa"], r["stress_GPa"], rtol=1e-4, atol=1e-3)


def _rattled_si():
    from ase.build import bulk

    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    atoms.rattle(0.05, seed=1)
    return atoms


def test_forces_and_stress_are_energy_derivatives(one_thread):
    # Elemental, so the canonical triplet frame has no element-swapping ties here and the
    # energy is smooth. Measured: 1.9e-3 eV/A and 6.6e-5 eV/A^3 (float32, h = 1e-2 / 1e-3).
    calc = make_calculator(MODEL, stress=True)
    atoms = _rattled_si()
    atoms.calc = calc
    forces, stress = atoms.get_forces(), atoms.get_stress(voigt=False)

    def energy(positions=None, cell=None):
        b = atoms.copy()
        b.calc = calc
        if cell is not None:
            b.set_cell(cell, scale_atoms=True)
        if positions is not None:
            b.set_positions(positions)
        return b.get_potential_energy()

    h, p = 1e-2, atoms.get_positions()
    for i, k in [(0, 0), (3, 1), (5, 2), (7, 0)]:
        plus, minus = p.copy(), p.copy()
        plus[i, k] += h
        minus[i, k] -= h
        assert -(energy(plus) - energy(minus)) / (2 * h) == pytest.approx(forces[i, k], abs=1e-2)

    h, c0, vol = 1e-3, atoms.get_cell().array, atoms.get_volume()
    for i, j in [(0, 0), (1, 2), (2, 1)]:
        eps = np.zeros((3, 3))
        eps[i, j] = h
        fd = (energy(cell=c0 @ (np.eye(3) + eps)) - energy(cell=c0 @ (np.eye(3) - eps))) / (2 * h) / vol
        assert fd == pytest.approx(stress[i, j], abs=5e-4)


def test_calculator_ignores_unwrapped_positions(one_thread):
    calc = make_calculator(MODEL, stress=True)
    atoms = _rattled_si()
    atoms.calc = calc
    e0, f0, s0 = atoms.get_potential_energy(), atoms.get_forces(), atoms.get_stress()
    shifted = atoms.copy()
    shifted.positions += np.array([[2, -1, 3]]) @ atoms.cell.array  # whole lattice vectors
    shifted.positions[0] += 3 * atoms.cell.array[1]
    shifted.calc = calc
    assert shifted.get_potential_energy() == pytest.approx(e0, abs=1e-4)
    np.testing.assert_allclose(shifted.get_forces(), f0, atol=1e-4)
    np.testing.assert_allclose(shifted.get_stress(), s0, atol=1e-6)


def test_calculator_stress_is_in_ev_per_cubic_angstrom(pot, one_thread):
    from ase import units

    from diep_pyg.ext.ase import Atoms2Graph

    atoms = _rattled_si()
    atoms.calc = make_calculator(MODEL, stress=True)
    g, lat, state = Atoms2Graph(pot.model.element_types, pot.model.cutoff).get_graph(atoms)
    _, _, s_gpa, _ = pot(g, lat, torch.tensor(state))
    np.testing.assert_allclose(atoms.get_stress(voigt=False), s_gpa.detach().numpy() * units.GPa,
                               rtol=1e-5, atol=1e-7)


def test_relaxed_silicon_lattice_constant():
    # The model gives 5.439 A (experiment: 5.431 A).
    from ase.build import bulk
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE

    atoms = bulk("Si", "diamond", a=5.50, cubic=True)
    atoms.calc = make_calculator(MODEL, stress=True)
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=0.01, steps=300)
    assert atoms.cell.lengths() == pytest.approx([5.439] * 3, abs=0.01)


# --- the whole pipeline on a synthetic MatPES-format file ------------------------------


def _write_synthetic_matpes(directory: Path, n: int = 40) -> tuple[Path, Path]:
    """MatPES-shaped records labelled by ASE's EMT, which is cheap and smooth."""
    from ase.build import bulk
    from ase.calculators.emt import EMT
    from pymatgen.io.ase import AseAtomsAdaptor

    rng = np.random.default_rng(0)
    records = []
    for i in range(n):
        el = ("Cu", "Al", "Ni")[i % 3]
        atoms = bulk(el, "fcc", a={"Cu": 3.61, "Al": 4.05, "Ni": 3.52}[el], cubic=i % 2 == 0)
        atoms = atoms.repeat((1, 1, 1 + i % 2))
        atoms.set_cell(atoms.cell.array * (1 + rng.normal(0, 0.02)), scale_atoms=True)
        atoms.rattle(0.05, seed=i)
        atoms.calc = EMT()
        stress_gpa = atoms.get_stress(voigt=True) / 0.006241509  # eV/A^3 -> GPa (Voigt)
        records.append({
            "structure": AseAtomsAdaptor.get_structure(atoms).as_dict(),
            "energy": float(atoms.get_potential_energy()),
            "forces": atoms.get_forces().tolist(),
            # MatPES stores VASP-convention stress in kbar; build_cache() applies * -0.1.
            "stress": (-10.0 * stress_gpa).tolist(),
            "functional": "r2SCAN",
        })
    data = directory / "MatPES-R2SCAN-2025.1.json.gz"
    with gzip.open(data, "wt") as f:
        json.dump(records, f)
    atoms_file = directory / "MatPES-R2SCAN-atoms.json.gz"
    with gzip.open(atoms_file, "wt") as f:
        json.dump([{"elements": [el], "energy": e} for el, e in
                   (("Cu", -0.2), ("Al", -0.1), ("Ni", -0.3))], f)
    return data, atoms_file


def test_pipeline_build_train_export_evaluate(tmp_path, one_thread):
    from diep_pyg import evaluate, matpes, train

    root = tmp_path / "data"
    root.mkdir()
    data, atoms_file = _write_synthetic_matpes(root)
    assert matpes.main(["build", "--root", str(root), "--json", str(data), "--atoms", str(atoms_file)]) == 0
    cache = root / "DIEPDataset"
    graphs = torch.load(cache / "pyg_graph.pt", weights_only=False)
    assert len(graphs) == 40 and not hasattr(graphs[0], "lattice")
    labels = json.loads((cache / "labels.json").read_text())
    assert np.asarray(labels["stresses"]).shape == (40, 3, 3)

    assert matpes.main(["folds", "--root", str(root), "--folds", "2"]) == 0
    with pytest.raises(SystemExit, match="already exist"):  # never redraws existing splits
        matpes.main(["folds", "--root", str(root), "--folds", "2"])
    meta = json.loads((root / "artifacts_full" / "fold1" / "meta.json").read_text())
    assert meta["seed"] == 43 and meta["split_sizes"] == {"train": 36, "val": 2, "test": 2}

    out = root / "runs_diep" / "diep_fold1"
    assert train.main(["--root", str(root), "--fold", "1", "--out-dir", str(out), "--max-epochs", "2",
                       "--batch-size", "4", "--accelerator", "cpu", "--num-workers", "0",
                       "--seed", "43", "--no-progress-bar"]) == 0
    run = json.loads((out / "metrics.json").read_text())
    ckpt = Path(run["best_checkpoint"])
    assert ckpt.exists() and np.isfinite(run["test"]["test_Total_Loss"])

    model_dir = tmp_path / "model"
    E.export(ckpt, root / "artifacts_full" / "fold1", model_dir, out / "metrics.json", "smoke")
    assert model_info(model_dir)["fold"] == 1

    # Same split, same batch size and order as train's own test pass, so evaluate's MAEs
    # and its logged-style RMSEs must reproduce the numbers train wrote.
    result = evaluate.evaluate(model_dir, matpes.DIEPConfig(root=root), root / "artifacts_full" / "fold1",
                               split="test", batch_size=4, predictions=tmp_path / "p.npz")
    t = run["test"]
    for key, logged in [("Energy_MAE", "test_Energy_MAE"), ("Force_MAE", "test_Force_MAE"),
                        ("Stress_MAE", "test_Stress_MAE"), ("Energy_RMSE_logged", "test_Energy_RMSE"),
                        ("Force_RMSE_logged", "test_Force_RMSE"), ("Stress_RMSE_logged", "test_Stress_RMSE")]:
        assert result[key] == pytest.approx(t[logged], rel=1e-5, abs=1e-7), key
    preds = np.load(tmp_path / "p.npz")
    assert len(preds["index"]) == 2 and preds["s_pred"].shape == (2, 3, 3)


if __name__ == "__main__" and "--regen" in sys.argv:
    torch.set_num_threads(1)
    REFERENCE.parent.mkdir(exist_ok=True)
    REFERENCE.write_text(json.dumps(_reference_predictions(load_potential(MODEL)), indent=1))
    print(f"wrote {REFERENCE}")
