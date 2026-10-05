"""Load an exported DIEP potential and wrap it as an ASE calculator.

    from diep_pyg.pretrained import load_potential, make_calculator

    atoms.calc = make_calculator()                  # the released diep_fold1, on CPU
    atoms.calc = make_calculator("models/diep_fold1", device="cuda", stress=False)

An exported model is a directory holding ``model.json``, ``model.pt`` and ``state.pt``,
written by ``Potential.save`` (see ``diep_pyg.export``). Loading needs neither Lightning
nor the training data. Every parameter is loaded strictly: a missing key raises instead of
leaving that weight at its random initialisation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from diep_pyg.apps.pes import Potential
from diep_pyg.ext.ase import PESCalculator

__all__ = ["DEFAULT_MODEL_DIR", "DIEPCalculator", "load_potential", "make_calculator", "model_info"]

#: The released model, resolved relative to this source tree. Set ``$DIEP_MODEL_DIR`` to
#: point elsewhere (for example when the package is installed without the models folder).
DEFAULT_MODEL_DIR = Path(os.environ.get(
    "DIEP_MODEL_DIR", Path(__file__).resolve().parent.parent / "models" / "diep_fold1"))

_FILES = ("model.json", "model.pt", "state.pt")


def load_potential(path: str | Path | None = None, device: str = "cpu", stress: bool = True):
    """Return the exported ``diep_pyg.apps.pes.Potential`` at ``path``, in eval mode.

    Args:
        path: model directory; defaults to the released ``models/diep_fold1``.
        device: torch device for the weights.
        stress: compute the stress (one extra gradient output per call). NVT dynamics and
            fixed-cell relaxations do not need it.
    """
    path = Path(path) if path is not None else DEFAULT_MODEL_DIR
    # Checked here rather than left to Potential.load, whose fallback for a path that is not
    # a complete model directory is to try downloading a model of that name from GitHub.
    missing = [f for f in _FILES if not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(f"{path} is not an exported DIEP model directory "
                                f"(missing {', '.join(missing)})")
    pot = Potential.load(path)
    pot.calc_stresses = bool(stress)
    pot.calc_hessian = False
    return pot.to(device).eval()


def model_info(path: str | Path | None = None) -> dict:
    """The provenance record shipped beside a model, or ``{}`` if there is none."""
    path = Path(path) if path is not None else DEFAULT_MODEL_DIR
    prov = path / "PROVENANCE.json"
    return json.loads(prov.read_text()) if prov.exists() else {}


class DIEPCalculator(PESCalculator):
    """``PESCalculator`` that builds its graph from positions wrapped into the cell.

    Nothing in an MD loop wraps positions, so atoms drift many cells out over a long run,
    and pymatgen's neighbour search sizes its image search from the bounding box of the
    input: cubically more work for the same graph (measured 0.23 -> 45.7 ms per step on
    liquid water). Wrapping is exact under PBC, since the graph carries ``pbc_offset``
    beside ``frac_coords``. ``self.atoms`` is reset to the caller's unwrapped atoms
    afterwards, or ASE's change detection would treat every later property request as a
    new geometry and recompute it. ``Atoms.wrap`` leaves non-periodic axes alone.
    """

    def calculate(self, atoms=None, properties=None, system_changes=None):
        wrapped = atoms.copy()
        wrapped.wrap()
        super().calculate(wrapped, properties, system_changes)
        self.atoms = atoms.copy()


def make_calculator(path: str | Path | None = None, device: str = "cpu", stress: bool = True):
    """An ASE calculator for the exported model at ``path`` (default: the released model).

    Stress is returned in eV/A^3, which is what ASE expects. (``PESCalculator`` on its own
    defaults to GPa, inherited from matgl; ASE's filters and barostats would then read it
    160x too large.)
    """
    pot = load_potential(path, device=device, stress=stress)
    return DIEPCalculator(potential=pot, stress_unit="eV/A3")
