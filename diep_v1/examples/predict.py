"""Energy, forces and stress for one structure with the released model.

    python examples/predict.py                         # built-in demo: rattled diamond Si
    python examples/predict.py POSCAR                  # any file ASE can read
    python examples/predict.py my.cif --relax          # relax positions and cell first
    python examples/predict.py my.cif --device cuda

Run from the release root (or after `pip install -e .`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ase import units
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.io import read
from ase.optimize import FIRE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diep_pyg.pretrained import make_calculator  # noqa: E402


def report(atoms, label: str) -> None:
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    s = atoms.get_stress(voigt=False) / units.GPa  # eV/A^3 -> GPa
    print(f"{label}: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
    print(f"  energy        {e:.6f} eV   ({e / len(atoms):.6f} eV/atom)")
    print(f"  max |F|       {np.linalg.norm(f, axis=1).max():.4f} eV/A")
    print(f"  pressure      {-np.trace(s) / 3:.4f} GPa")
    print("  stress (GPa)  " + np.array2string(s, precision=4, suppress_small=True).replace("\n", "\n" + " " * 16))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("structure", nargs="?", default=None, help="structure file; omit for the demo")
    p.add_argument("--model", default=None, help="exported model dir (default: models/diep_fold1)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--relax", action="store_true", help="relax positions and cell (FIRE)")
    p.add_argument("--fmax", type=float, default=0.05)
    args = p.parse_args(argv)

    if args.structure:
        atoms = read(args.structure)
    else:
        atoms = bulk("Si", "diamond", a=5.43, cubic=True)
        atoms.rattle(0.05, seed=0)
    atoms.calc = make_calculator(args.model, device=args.device, stress=True)
    report(atoms, "input")

    if args.relax:
        FIRE(FrechetCellFilter(atoms), logfile="-").run(fmax=args.fmax, steps=500)
        report(atoms, "relaxed")
        print(f"  cell lengths  {atoms.cell.lengths().round(4)} A, volume {atoms.get_volume():.3f} A^3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
