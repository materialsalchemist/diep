"""Short NVT molecular dynamics with the released model.

    python examples/md.py                                    # 64-atom Si, 600 K, 200 fs
    python examples/md.py --structure my.cif --supercell 2 --temperature 1000 --steps 5000
    python examples/md.py --device cuda --traj md.traj

Langevin thermostat, fixed cell. Each log line includes the largest force on any atom:
the canonical triplet frame this model uses can produce spurious forces of tens of eV/A on
nearly straight three-atom chains in float32 (see README, "Known limitations"), so a
sudden jump in max |F| is worth a look before trusting a trajectory.

Run from the release root (or after `pip install -e .`).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from ase import units
from ase.build import bulk
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diep_pyg.pretrained import make_calculator  # noqa: E402

SPIKE = 10.0  # eV/A; flagged in the log


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--structure", default=None, help="structure file; default diamond Si")
    p.add_argument("--supercell", type=int, default=2, help="repeat along each axis")
    p.add_argument("--model", default=None, help="exported model dir (default: models/diep_fold1)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--temperature", type=float, default=600.0, help="K")
    p.add_argument("--timestep", type=float, default=1.0, help="fs")
    p.add_argument("--friction", type=float, default=0.01, help="1/fs")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--traj", default=None, help="write an ASE trajectory here")
    args = p.parse_args(argv)

    atoms = read(args.structure) if args.structure else bulk("Si", "diamond", a=5.43, cubic=True)
    atoms = atoms.repeat(args.supercell)
    # No stress needed at fixed cell; skipping it saves a gradient output per step.
    atoms.calc = make_calculator(args.model, device=args.device, stress=False)

    rng = np.random.default_rng(args.seed)
    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature, rng=rng)
    Stationary(atoms)
    dyn = Langevin(atoms, args.timestep * units.fs, temperature_K=args.temperature,
                   friction=args.friction / units.fs, rng=rng)
    if args.traj:
        dyn.attach(Trajectory(args.traj, "w", atoms), interval=args.log_every)

    t0 = time.time()
    print(f"{atoms.get_chemical_formula()}: {len(atoms)} atoms, {args.temperature:.0f} K, "
          f"dt {args.timestep} fs, {args.steps} steps")
    print(f"{'step':>7} {'t (ps)':>8} {'T (K)':>8} {'Epot (eV/atom)':>15} {'max|F| (eV/A)':>14} {'s/step':>8}")

    def log():
        f = np.linalg.norm(atoms.get_forces(), axis=1).max()
        n = dyn.nsteps
        rate = (time.time() - t0) / max(n, 1)
        flag = "  SPIKE" if f > SPIKE else ""
        print(f"{n:>7} {n * args.timestep / 1000:>8.3f} {atoms.get_temperature():>8.1f} "
              f"{atoms.get_potential_energy() / len(atoms):>15.5f} {f:>14.3f} {rate:>8.3f}{flag}",
              flush=True)

    dyn.attach(log, interval=args.log_every)  # ASE also fires observers at step 0
    dyn.run(args.steps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
