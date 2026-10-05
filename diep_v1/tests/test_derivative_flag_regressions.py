"""Regression tests for two defects fixed on 2026-09-28 (second review pass).

Neither shows up on the default configuration, which is why the existing numerical checks
(forces vs finite differences, Hessian symmetry, batching) all passed while they were live.

1. ``Potential.forward`` computed the energy gradient only under ``calc_forces``, then let
   ``calc_hessian``, ``calc_stresses`` and ``debug_mode`` index the result unconditionally.
   Asking for a Hessian or a stress *without* forces raised ``TypeError`` on ``None``, and
   ``debug_mode`` without stresses raised ``IndexError`` off the end of a one-element list.
   ``g.pos`` was additionally only made differentiable under ``calc_forces``, so the Hessian
   path had no graph to differentiate even once the indexing was fixed.

2. ``IOMixIn.load`` called ``load_state_dict(..., strict=False)``, so a checkpoint whose keys
   did not line up loaded with no exception and no warning, leaving the unmatched parameters
   at their random initialisation. The model then predicted confidently from untrained
   weights -- the same failure mode recorded for ``matgl.load_model``.
"""

from __future__ import annotations

import itertools

import pytest
import torch
from pymatgen.core import Lattice, Structure

from diep_pyg.apps.pes import Potential
from diep_pyg.graph.converters import Structure2Graph
from diep_pyg.models import DIEP

ELEMENTS = ("Si",)
CUTOFF = 5.0
THREEBODY_CUTOFF = 4.0

SI = Structure(
    Lattice([[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]]),
    ["Si"] * 2,
    [[0, 0, 0], [0.25, 0.25, 0.25]],
)


def _graph():
    """A fresh graph each time: ``Potential.forward`` writes positions onto it."""
    converter = Structure2Graph(element_types=ELEMENTS, cutoff=CUTOFF)
    g, lattice, _ = converter.get_graph(SI)
    return g, lattice


def _model() -> DIEP:
    torch.manual_seed(0)
    return DIEP(
        element_types=ELEMENTS,
        integral_mode="sum",
        cutoff=CUTOFF,
        threebody_cutoff=THREEBODY_CUTOFF,
        nblocks=1,
        units=8,
        dim_node_embedding=8,
        dim_edge_embedding=8,
        is_intensive=False,
    )


@pytest.mark.parametrize(
    ("calc_forces", "calc_stresses", "calc_hessian"),
    list(itertools.product([True, False], repeat=3)),
)
def test_every_derivative_flag_combination_runs(calc_forces, calc_stresses, calc_hessian):
    """All eight flag combinations must return, not raise on a missing gradient."""
    g, lattice = _graph()
    potential = Potential(
        _model(), calc_forces=calc_forces, calc_stresses=calc_stresses, calc_hessian=calc_hessian
    )
    energies, forces, stresses, hessian = potential(g, lattice)

    assert torch.isfinite(energies).all()
    n = 3 * len(SI)
    # A disabled quantity keeps its placeholder shape; an enabled one is fully populated.
    assert forces.shape == (len(SI), 3) if calc_forces else forces.shape == (1,)
    assert hessian.shape == (n, n) if calc_hessian else hessian.shape == (1,)
    if calc_stresses:
        assert stresses.shape == (3, 3)
    else:
        assert stresses.shape == (1,)


def test_stress_and_hessian_do_not_depend_on_calc_forces():
    """Turning forces off must not change the stress or the Hessian it is bundled with."""
    model = _model()

    g, lattice = _graph()
    _, _, stress_ref, hessian_ref = Potential(
        model, calc_forces=True, calc_stresses=True, calc_hessian=True
    )(g, lattice)

    g, lattice = _graph()
    _, _, stress_only, _ = Potential(model, calc_forces=False, calc_stresses=True)(g, lattice)

    g, lattice = _graph()
    _, _, _, hessian_only = Potential(
        model, calc_forces=False, calc_stresses=False, calc_hessian=True
    )(g, lattice)

    assert torch.equal(stress_only, stress_ref)
    assert torch.equal(hessian_only, hessian_ref)


def test_hessian_without_forces_is_populated_and_symmetric():
    """The Hessian is never enforced symmetric, so symmetry is real evidence it is right."""
    g, lattice = _graph()
    _, _, _, hessian = Potential(_model(), calc_forces=False, calc_hessian=True)(g, lattice)

    n = 3 * len(SI)
    assert hessian.shape == (n, n)
    assert torch.isfinite(hessian).all()
    assert not torch.all(hessian == 0), "Hessian was never filled in"
    assert torch.allclose(hessian, hessian.T, atol=1e-6)


def test_debug_mode_without_stresses():
    """``debug_mode`` used to read a stress gradient that was never requested."""
    g, lattice = _graph()
    energies, pos_grad, stress_grad = Potential(
        _model(), calc_forces=True, calc_stresses=False, debug_mode=True
    )(g, lattice)

    assert torch.isfinite(energies).all()
    assert pos_grad.shape == (len(SI), 3)
    assert stress_grad is None  # not requested, so not fabricated


def test_load_rejects_a_checkpoint_with_missing_weights(tmp_path):
    """A key mismatch must raise, not silently leave parameters randomly initialised."""
    model = _model()
    model.save(tmp_path)

    state_path = tmp_path / "state.pt"
    state = torch.load(state_path, weights_only=False)
    renamed = {key.replace("graph_layers", "renamed_layers"): value for key, value in state.items()}
    assert renamed != state, "fixture failed to rename anything"
    torch.save(renamed, state_path)

    with pytest.raises(ValueError, match="missing"):
        DIEP.load(tmp_path)


def test_load_round_trip_is_exact(tmp_path):
    """The strict check must not reject an honest checkpoint."""
    model = _model()
    before = model.predict_structure(SI)
    model.save(tmp_path)

    reloaded = DIEP.load(tmp_path)
    assert torch.equal(reloaded.predict_structure(SI), before)


def test_load_tolerates_unused_keys(tmp_path):
    """Extra keys cannot leave a parameter uninitialised, so they warn rather than raise."""
    model = _model()
    before = model.predict_structure(SI)
    model.save(tmp_path)

    state_path = tmp_path / "state.pt"
    state = torch.load(state_path, weights_only=False)
    state["a.future.head.weight"] = torch.zeros(3)
    torch.save(state, state_path)

    with pytest.warns(UserWarning, match="not used by"):
        reloaded = DIEP.load(tmp_path)
    assert torch.equal(reloaded.predict_structure(SI), before)
