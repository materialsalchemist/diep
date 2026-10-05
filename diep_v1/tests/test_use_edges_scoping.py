"""Regression tests for the triplet on/off switch, added 2026-09-29.

``use_edges`` / ``use_triplets`` select whether the three-body channel runs at all, which is
the single variable a paired DIEP run is built to isolate. Two defects made that switch lie:

* ``Potential.__init__`` wrote ``use_edges`` straight into the model and never restored it.
  The model is shared state, so building a triplet-off potential and then a triplet-on one
  from the same model left *both* running triplet-off -- the second was constructed with
  ``use_edges=None``, documented as "keep whatever the model was built with", after the first
  had already flipped the model's flag. Same shapes, same loss scale, no warning.
* ``Potential.forward`` re-applied ``use_edges`` on every call but not ``use_triplets``, so
  the two attributes -- separate copies of one fact -- drifted apart. ``DIEP.forward`` reads
  ``use_edges``; ``diep_pyg/train.py`` writes ``use_triplets`` into the run manifest.
  A drifted pair means the recorded metadata describes an arm that was not the one run.

Each test below was confirmed to fail before the fix.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from diep_pyg.apps.pes import Potential
from diep_pyg.graph.converters import Structure2Graph
from diep_pyg.models._diep import DIEP

_ELEMENTS = ("O", "Si")


def _structure() -> Structure:
    return Structure(
        Lattice.cubic(5.43),
        ["Si", "Si", "O"],
        [[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.0]],
    )


def _model(**kwargs) -> DIEP:
    torch.manual_seed(0)
    return DIEP(
        element_types=_ELEMENTS,
        nblocks=1,
        is_intensive=False,
        integral_mode="sum",
        cutoff=5.0,
        threebody_cutoff=4.0,
        **kwargs,
    )


def _graph(structure: Structure):
    converter = Structure2Graph(element_types=_ELEMENTS, cutoff=5.0)
    graph, lattice, state_attr = converter.get_graph(structure)
    return graph, lattice, torch.tensor(state_attr)


def _energy(potential: Potential, structure: Structure) -> float:
    graph, lattice, state_attr = _graph(structure)
    return float(potential(graph, lattice, state_attr)[0])


def test_potential_does_not_mutate_the_model_it_wraps():
    """Constructing a triplet-off Potential must leave the model's own flag untouched."""
    model = _model(use_triplets=True)
    Potential(model, calc_forces=False, calc_stresses=False, use_edges=False)
    assert model.use_edges is True
    assert model.use_triplets is True


def test_paired_potentials_over_one_model_stay_independent():
    """The triplet-on / triplet-off pair of a paired run must give different energies.

    Before the fix the triplet-off potential flipped the shared model's flag permanently, so
    the triplet-on arm -- built afterwards with the default ``use_edges=None`` -- silently
    ran triplet-off too and both arms returned the same number.
    """
    structure = _structure()
    model = _model(use_triplets=True)

    off = Potential(model, calc_forces=False, calc_stresses=False, use_edges=False)
    on = Potential(model, calc_forces=False, calc_stresses=False)

    e_off = _energy(off, structure)
    e_on = _energy(on, structure)

    assert not np.isclose(e_on, e_off, atol=1e-8), (
        f"triplet-on and triplet-off arms returned the same energy ({e_on}); the shared "
        "model was reconfigured by the other potential"
    )
    # Construction order must not matter either.
    assert np.isclose(_energy(on, structure), e_on, atol=1e-12)
    assert np.isclose(_energy(off, structure), e_off, atol=1e-12)


def test_forward_restores_the_model_flag():
    """An override lasts exactly one forward pass."""
    structure = _structure()
    model = _model(use_triplets=True)
    potential = Potential(model, calc_forces=False, calc_stresses=False, use_edges=False)

    _energy(potential, structure)

    assert model.use_edges is True, "use_edges was left overridden after the forward pass"
    assert model.use_triplets is True


def test_use_edges_none_respects_the_models_own_setting():
    """``use_edges=None`` means 'do not override', in both directions."""
    structure = _structure()

    model_off = _model(use_triplets=False)
    plain = Potential(model_off, calc_forces=False, calc_stresses=False)
    forced_off = Potential(model_off, calc_forces=False, calc_stresses=False, use_edges=False)
    assert np.isclose(_energy(plain, structure), _energy(forced_off, structure), atol=1e-12)

    model_on = _model(use_triplets=True)
    plain_on = Potential(model_on, calc_forces=False, calc_stresses=False)
    forced_on = Potential(model_on, calc_forces=False, calc_stresses=False, use_edges=True)
    assert np.isclose(_energy(plain_on, structure), _energy(forced_on, structure), atol=1e-12)


def test_use_edges_and_use_triplets_cannot_drift_apart():
    """The two names are one flag, so writing either moves both.

    ``DIEP.forward`` reads ``use_edges`` while run manifests record ``use_triplets``. When
    they were separate attributes, ``Potential.forward`` wrote only the first and the
    manifest then described an arm that had not been run.
    """
    model = _model(use_triplets=True)

    model.use_edges = False
    assert model.use_triplets is False, "writing use_edges left use_triplets stale"

    model.use_triplets = True
    assert model.use_edges is True, "writing use_triplets left use_edges stale"


def test_flag_is_restored_even_when_the_forward_pass_raises():
    """A failed evaluation must not leave the model reconfigured for every later call."""
    model = _model(use_triplets=True)
    potential = Potential(model, calc_forces=False, calc_stresses=False, use_edges=False)

    with pytest.raises(Exception):
        # No graph at all: the model raises part-way through, inside the override.
        potential(object(), torch.eye(3), None)

    assert model.use_edges is True
    assert model.use_triplets is True


def test_use_edges_on_a_model_without_the_attribute_is_rejected():
    """An override that could never take effect fails loudly at construction."""

    class Bare(torch.nn.Module):
        cutoff = 5.0

    with pytest.raises(ValueError, match="use_edges"):
        Potential(Bare(), calc_forces=False, calc_stresses=False, use_edges=False)
