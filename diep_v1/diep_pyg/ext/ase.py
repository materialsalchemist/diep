"""Interfaces to the Atomic Simulation Environment package for dynamic simulations.

PyG counterpart of :mod:`diep.ext.ase`. The only backend-specific pieces are
``Atoms2Graph`` (builds a :class:`diep_pyg.graph.compute.DIEPData` instead of a DGL
graph) and the graph-construction/potential-call lines inside
``PESCalculator.calculate()``; everything else (optimizer/ensemble dispatch,
trajectory recording) is plain ASE machinery, copied unchanged.
"""

from __future__ import annotations

import collections
import contextlib
import io
import pickle
import sys
from enum import Enum
from typing import TYPE_CHECKING, Literal

import ase.optimize as opt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.filters import FrechetCellFilter
from ase.md import Langevin
from ase.md.andersen import Andersen
from ase.md.bussi import Bussi
from ase.md.nose_hoover_chain import IsotropicMTKNPT, NoseHooverChainNVT
from ase.md.npt import NPT
from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.stress import full_3x3_to_voigt_6_stress
from pymatgen.core.structure import Molecule, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.optimization.neighbors import find_points_in_spheres

from diep_pyg.graph.converters import GraphConverter

if TYPE_CHECKING:
    from typing import Any

    from ase.optimize.optimize import Optimizer

    from diep_pyg.apps.pes import Potential
    from diep_pyg.graph.compute import DIEPData


class OPTIMIZERS(Enum):
    """An enumeration of optimizers for used in."""

    fire = opt.fire.FIRE
    bfgs = opt.bfgs.BFGS
    lbfgs = opt.lbfgs.LBFGS
    lbfgslinesearch = opt.lbfgs.LBFGSLineSearch
    mdmin = opt.mdmin.MDMin
    scipyfmincg = opt.sciopt.SciPyFminCG
    scipyfminbfgs = opt.sciopt.SciPyFminBFGS
    bfgslinesearch = opt.bfgslinesearch.BFGSLineSearch


class Atoms2Graph(GraphConverter):
    """Construct a PyG graph from ASE Atoms."""

    def __init__(
        self,
        element_types: tuple[str, ...],
        cutoff: float = 5.0,
    ):
        """Init Atoms2Graph from element types and cutoff radius.

        Args:
            element_types: List of elements present in dataset for graph conversion. This ensures all graphs are
                constructed with the same dimensionality of features.
            cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, atoms: Atoms) -> tuple[DIEPData, torch.Tensor, list | np.ndarray]:
        """Get a PyG graph from an input Atoms.

        Args:
            atoms: Atoms object.

        Returns:
            g: PyG graph
            state_attr: state features
        """
        numerical_tol = 1.0e-8
        element_types = self.element_types
        cart_coords = atoms.get_positions()
        # Periodic along *any* axis means the real cell and a neighbour search over images.
        # This branch used to test `atoms.pbc.all()`, which sent a slab (pbc=[True, True,
        # False]) down the isolated-molecule path: identity lattice, zero image offsets,
        # Cartesian coordinates written into the `frac_coords` slot, and only in-cell
        # neighbours. On a 3-layer Si slab that dropped 168 edges to 18 -- 89% of the
        # neighbourhood -- with no warning, and left the stress divided by a unit volume
        # rather than the real one (120.08 A^3), i.e. wrong by ~120x.
        #
        # `find_points_in_spheres` takes a per-axis pbc array, so mixed periodicity needs no
        # special case: it searches images only along the periodic axes, giving 136 edges for
        # that same slab -- between the 168 fully-periodic and 18 isolated counts.
        periodic = np.asarray(atoms.pbc, dtype=bool)
        if periodic.any():
            lattice_matrix = np.array(atoms.get_cell())
            if abs(float(np.linalg.det(lattice_matrix))) < numerical_tol:
                raise ValueError(
                    "Atoms is periodic along at least one axis but its cell is singular, so "
                    "fractional coordinates and the stress volume are undefined. Give it a "
                    "full cell, or set atoms.pbc = False for an isolated system."
                )
            # Note: np.int64 explicitly, or this defaults to long and fails on Windows.
            pbc = np.array(periodic, dtype=np.int64)
            src_id, dst_id, images, bond_dist = find_points_in_spheres(
                cart_coords,
                cart_coords,
                r=self.cutoff,
                pbc=pbc,
                lattice=lattice_matrix,
                tol=numerical_tol,
            )
            exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
            src_id, dst_id, images, bond_dist = (
                src_id[exclude_self],
                dst_id[exclude_self],
                images[exclude_self],
                bond_dist[exclude_self],
            )
            # Fractional coordinates against the real cell, unwrapped: `pbc_offset` already
            # carries each bond's image, so wrapping here would double-count the shift.
            coords = atoms.get_scaled_positions(wrap=False)
            lattice_arg = [lattice_matrix]
        else:
            lattice_matrix = np.expand_dims(np.identity(3), axis=0)
            dist = np.linalg.norm(cart_coords[:, None, :] - cart_coords[None, :, :], axis=-1)
            adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(len(cart_coords), dtype=np.bool_)
            adj = adj.tocoo()
            src_id = adj.row
            dst_id = adj.col
            images = np.zeros((len(adj.row), 3))
            coords = cart_coords
            lattice_arg = lattice_matrix

        g, lat, state_attr = super().get_graph_from_processed_structure(
            atoms,
            src_id,
            dst_id,
            images,
            lattice_arg,
            element_types,
            coords,
            is_atoms=True,
        )

        return g, lat, state_attr


class PESCalculator(Calculator):
    """Potential calculator for ASE."""

    implemented_properties = ["energy", "free_energy", "forces", "stress", "hessian", "magmoms"]  # noqa:RUF012

    def __init__(
        self,
        potential: Potential,
        state_attr: torch.Tensor | None = None,
        stress_unit: Literal["eV/A3", "GPa"] = "GPa",
        stress_weight: float = 1.0,
        use_voigt: bool = False,
        **kwargs,
    ):
        """
        Init PESCalculator with a Potential from diep.

        Args:
            potential (Potential): diep_pyg.apps.pes.Potential
            state_attr (tensor): State attribute
            compute_stress (bool): whether to calculate the stress
            stress_unit (str): stress unit. Default: "GPa"
            stress_weight (float): conversion factor from GPa to eV/A^3, if it is set to 1.0, the unit is in GPa
            use_voigt (bool): whether the voigt notation is used for stress output
            **kwargs: Kwargs pass through to super().__init__().
        """
        super().__init__(**kwargs)
        self.potential = potential
        self.compute_stress = potential.calc_stresses
        self.compute_hessian = potential.calc_hessian
        self.compute_magmom = potential.calc_magmom

        self.graph_converter = Atoms2Graph(potential.model.element_types, potential.model.cutoff)

        # Handle stress unit conversion
        if stress_unit == "eV/A3":
            conversion_factor = units.GPa / (units.eV / units.Angstrom**3)  # Conversion factor from GPa to eV/A^3
        elif stress_unit == "GPa":
            conversion_factor = 1.0  # No conversion needed if stress is already in GPa
        else:
            raise ValueError(f"Unsupported stress_unit: {stress_unit}. Must be 'GPa' or 'eV/A3'.")

        self.stress_weight = stress_weight * conversion_factor
        self.state_attr = state_attr
        self.element_types = potential.model.element_types  # type: ignore
        self.cutoff = potential.model.cutoff
        self.use_voigt = use_voigt

    def _potential_device(self):
        """Device the potential's tensors live on, or None if it holds none.

        Reads parameters and buffers of the whole potential, not just ``potential.model``,
        so submodules like ``element_refs`` / ``repuls`` are taken into account.
        """
        for tensor in self.potential.parameters():
            return tensor.device
        for tensor in self.potential.buffers():
            return tensor.device
        return None

    def calculate(  # type:ignore[override]
        self,
        atoms: Atoms,
        properties: list | None = None,
        system_changes: list | None = None,
    ):
        """
        Perform calculation for an input Atoms.

        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        """
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        graph, lattice, state_attr_default = self.graph_converter.get_graph(atoms)
        # Move the freshly built (CPU) graph to wherever the potential already lives, rather
        # than demoting the potential to CPU. The previous code called
        # `self.potential.model.to("cpu")` and never restored the device, so the first
        # calculate() on a GPU-resident potential permanently relocated it and every
        # subsequent MD/relaxation step ran on CPU. It also moved only `model`, leaving
        # `element_refs` / `repuls` on the GPU and mismatched.
        device = self._potential_device()
        if device is not None and device.type != "cpu":
            graph = graph.to(device)
            lattice = lattice.to(device)
        state_attr = self.state_attr if self.state_attr is not None else state_attr_default
        if isinstance(state_attr, torch.Tensor) and device is not None:
            state_attr = state_attr.to(device)
        calc_result = self.potential(graph, lattice, state_attr)
        self.results.update(
            energy=calc_result[0].detach().cpu().numpy().item(),
            free_energy=calc_result[0].detach().cpu().numpy().item(),
            forces=calc_result[1].detach().cpu().numpy(),
        )
        if self.compute_stress:
            stresses_np = (
                full_3x3_to_voigt_6_stress(calc_result[2].detach().cpu().numpy())
                if self.use_voigt
                else calc_result[2].detach().cpu().numpy()
            )
            self.results.update(stress=stresses_np * self.stress_weight)
        if self.compute_hessian:
            self.results.update(hessian=calc_result[3].detach().cpu().numpy())
        if self.compute_magmom:
            self.results.update(magmoms=calc_result[4].detach().cpu().numpy())


class Relaxer:
    """Relaxer is a class for structural relaxation."""

    def __init__(
        self,
        potential: Potential,
        state_attr: torch.Tensor | None = None,
        optimizer: Optimizer | str = "FIRE",
        relax_cell: bool = True,
        stress_weight: float = 1 / 160.21766208,
    ):
        """
        Args:
            potential (Potential): a diep potential, a str path to a saved model or a short name for a saved model
            state_attr (torch.Tensor): State attr.
            optimizer (str or ase Optimizer): the optimization algorithm.
            Defaults to "FIRE"
            relax_cell (bool): whether to relax the lattice cell
            stress_weight (float): conversion factor from GPa to eV/A^3.
        """
        self.optimizer: Optimizer = OPTIMIZERS[optimizer.lower()].value if isinstance(optimizer, str) else optimizer
        self.calculator = PESCalculator(
            potential=potential,
            state_attr=state_attr,
            stress_weight=stress_weight,  # type: ignore
        )
        self.relax_cell = relax_cell
        self.ase_adaptor = AseAtomsAdaptor()

    def relax(
        self,
        atoms: Atoms | Structure | Molecule,
        fmax: float = 0.1,
        steps: int = 500,
        traj_file: str | None = None,
        interval: int = 1,
        verbose: bool = False,
        ase_cellfilter: Literal["Frechet", "Exp"] = "Frechet",
        params_asecellfilter: dict | None = None,
        **kwargs,
    ):
        """
        Relax an input Atoms.

        Args:
            atoms (Atoms | Structure | Molecule): the atoms for relaxation
            fmax (float): total force tolerance for relaxation convergence.
            Here fmax is a sum of force and stress forces
            steps (int): max number of steps for relaxation
            traj_file (str): the trajectory file for saving
            interval (int): the step interval for saving the trajectories
            verbose (bool): Whether to have verbose output.
            ase_cellfilter (literal): which filter is used for variable cell relaxation. Default is Frechet.
            params_asecellfilter (dict): Parameters to be passed to FrechetCellFilter. Allows
                setting of constant pressure or constant volume relaxations, for example. Refer to
                https://wiki.fysik.dtu.dk/ase/ase/filters.html#FrechetCellFilter for more information.
            **kwargs: Kwargs pass-through to optimizer.
        """
        if isinstance(atoms, Structure | Molecule):
            atoms = self.ase_adaptor.get_atoms(atoms)
        atoms.set_calculator(self.calculator)
        stream = sys.stdout if verbose else io.StringIO()
        params_asecellfilter = params_asecellfilter or {}
        with contextlib.redirect_stdout(stream):
            obs = TrajectoryObserver(atoms)
            if self.relax_cell:
                atoms = (
                    FrechetCellFilter(atoms, **params_asecellfilter)  # type:ignore[assignment]
                )

            optimizer = self.optimizer(atoms, **kwargs)  # type:ignore[operator]
            optimizer.attach(obs, interval=interval)
            optimizer.run(fmax=fmax, steps=steps)
            # Capture the final configuration only if the attached observer did not already
            # fire on it. With the default interval=1 it always does, so the unconditional
            # `obs()` recorded the last frame twice -- `energies[-1] == energies[-2]` with
            # identical positions -- and any per-frame statistic over the trajectory (mean
            # energy, step count, MSD) double-weighted the final configuration. The call still
            # matters for interval > 1, where the last step need not land on a recorded frame.
            if optimizer.nsteps % interval != 0:
                obs()
        if traj_file is not None:
            obs.save(traj_file)

        if isinstance(atoms, FrechetCellFilter):
            atoms = atoms.atoms

        final_structure: Structure | Molecule
        if isinstance(atoms, Atoms):
            if np.array(atoms.pbc).any():
                final_structure = self.ase_adaptor.get_structure(atoms)
            else:
                final_structure = self.ase_adaptor.get_molecule(atoms)
        elif isinstance(atoms, Structure | Molecule):
            final_structure = atoms
        else:
            raise TypeError(f"Unsupported atoms type: {type(atoms)}")

        return {
            "final_structure": final_structure,  # type:ignore[arg-type]
            "trajectory": obs,
        }


class TrajectoryObserver(collections.abc.Sequence):
    """Trajectory observer is a hook in the relaxation process that saves the
    intermediate structures.
    """

    def __init__(self, atoms: Atoms) -> None:
        """
        Init the Trajectory Observer from a Atoms.

        Args:
            atoms (Atoms): Structure to observe.
        """
        self.atoms = atoms
        self.energies: list[float] = []
        self.forces: list[np.ndarray] = []
        self.stresses: list[np.ndarray] = []
        self.atom_positions: list[np.ndarray] = []
        self.cells: list[np.ndarray] = []

    def __call__(self) -> None:
        """Record one frame.

        Every list gains exactly one entry per frame, ``None`` where the quantity is not
        available. Stress used to be appended only when the calculator computed it and the cell
        only for a periodic system, while ``__getitem__`` and ``as_pandas`` index all five
        lists by the same frame number: with ``calc_stresses=False`` a relaxation left
        ``len(energies) == 2`` against ``len(stresses) == 0``, so ``obs[0]`` raised
        ``IndexError`` and ``as_pandas()`` raised ``ValueError: All arrays must be of the same
        length``. Keeping the lists parallel is what those two accessors already assume.
        """
        self.energies.append(float(self.atoms.get_potential_energy()))
        self.forces.append(self.atoms.get_forces())
        self.stresses.append(self.atoms.get_stress() if self.atoms.calc.compute_stress else None)
        self.atom_positions.append(self.atoms.get_positions())
        self.cells.append(self.atoms.get_cell()[:] if self.atoms.pbc.any() else None)

    def __getitem__(self, item):
        return self.energies[item], self.forces[item], self.stresses[item], self.cells[item], self.atom_positions[item]

    def __len__(self):
        return len(self.energies)

    def as_pandas(self) -> pd.DataFrame:
        """Returns: DataFrame of energies, forces, stresses, cells and atom_positions."""
        return pd.DataFrame(
            {
                "energies": self.energies,
                "forces": self.forces,
                "stresses": self.stresses,
                "cells": self.cells,
                "atom_positions": self.atom_positions,
            }
        )

    def save(self, filename: str) -> None:
        """Save the trajectory to file.

        Args:
            filename (str): filename to save the trajectory.
        """
        out = {
            "energy": self.energies,
            "forces": self.forces,
            "stresses": self.stresses,
            "atom_positions": self.atom_positions,
            "cell": self.cells,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out, file)


class MolecularDynamics:
    """Molecular dynamics class."""

    def __init__(
        self,
        atoms: Atoms,
        potential: Potential,
        state_attr: torch.Tensor | None = None,
        stress_weight: float = 1.0,
        ensemble: Literal[
            "nve",
            "nvt",
            "nvt_langevin",
            "nvt_andersen",
            "nvt_bussi",
            "nvt_nose_hoover_chain",
            "npt",
            "npt_berendsen",
            "npt_nose_hoover",
            "npt_nose_hoover_chain",
        ] = "nvt",
        temperature: int = 300,
        timestep: float = 1.0,
        pressure: float = 1.01325 * units.bar,
        taut: float | None = None,
        taup: float | None = None,
        friction: float = 1.0e-2,
        andersen_prob: float = 1.0e-2,
        ttime: float = 25.0,
        ptime: float = 75.0,
        bulk_modulus: float = 0.6,
        pfactor: float | None = None,
        external_stress: float | np.ndarray | None = None,
        compressibility_au: float | None = None,
        trajectory: Any = None,
        logfile: str | None = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
        mask: tuple | np.ndarray | None = None,
    ):
        """
        Init the MD simulation.

        Args:
            atoms (Atoms): atoms to run the MD
            potential (Potential): potential for calculating the energy, force,
            stress of the atoms
            state_attr (torch.Tensor): State attr.
            stress_weight (float): conversion factor from GPa to eV/A^3
            ensemble (str): choose from "nve", "nvt", "nvt_langevin", "nvt_andersen", "nvt_bussi",
            "npt", "npt_berendsen", "npt_nose_hoover"
            temperature (float): temperature for MD simulation, in K
            timestep (float): time step in fs
            pressure (float): pressure in eV/A^3
            taut (float): time constant for temperature coupling
            taup (float): time constant for pressure coupling
            friction (float): friction coefficient for nvt_langevin, typically set to 1e-4 to 1e-2
            andersen_prob (float): random collision probability for nvt_andersen, typically set to 1e-4 to 1e-1
            ttime (float): Characteristic timescale of the thermostat, in fs.
            ptime (float): Characteristic barostat timescale, in fs. Used with
                ``bulk_modulus`` to build ASE's ``pfactor``, which it defines as
                ``ptime**2 * B``. Only ``npt_nose_hoover`` uses it.
            bulk_modulus (float): Bulk modulus of the material in eV/A^3, the ``B`` in
                ``pfactor = ptime**2 * B``. The 0.6 default is a rough value for a hard
                solid; a soft or molecular system wants a smaller one.
            pfactor (float): ASE's barostat constant, ``ptime**2 * B``, in ASE internal
                units. Leave as None to have it built from ``ptime`` and ``bulk_modulus``;
                pass a number to set it directly, in which case both are ignored.
            external_stress (float): The external stress in eV/A^3.
                Either 3x3 tensor,6-vector or a scalar representing pressure
            compressibility_au (float): compressibility of the material in A^3/eV
            trajectory (str or Trajectory): Attach trajectory object
            logfile (str): open this file for recording MD outputs
            loginterval (int): write to log file every interval steps
            append_trajectory (bool): Whether to append to prev trajectory.
            mask (np.array): either a tuple of 3 numbers (0 or 1) or a symmetric 3x3 array indicating,
                which strain values may change for NPT simulations.
        """
        if isinstance(atoms, Structure | Molecule):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        self.atoms = atoms
        self.atoms.set_calculator(
            PESCalculator(potential=potential, state_attr=state_attr, stress_unit="eV/A3", stress_weight=stress_weight)
        )

        if taut is None:
            taut = 100 * timestep * units.fs
        if taup is None:
            taup = 1000 * timestep * units.fs

        # ASE defines `pfactor = ptime**2 * B`, with ptime in ASE time units and B a bulk
        # modulus in eV/A^3. The old default passed `75.0**2 * units.fs` straight through,
        # which was wrong twice over: the time factor has to be *squared* (units.fs**2, not
        # units.fs) and the bulk modulus was missing entirely. That gave pfactor = 552.53
        # where "ptime = 75 fs, B = 0.6 eV/A^3" is 32.56 -- a factor of 16.97, so the
        # effective barostat timescale was ~309 fs rather than the documented 75 fs, and the
        # barostat was roughly 17x too stiff. The adjacent `ttime * units.fs` was correct all
        # along, which is what made the pfactor line look right by association.
        if pfactor is None:
            pfactor = (ptime * units.fs) ** 2 * bulk_modulus

        # The Berendsen barostats multiply by `compressibility_au`, so None is not a usable
        # default: `ensemble="npt"` and `"npt_berendsen"` both raised
        # `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'` with no extra
        # arguments at all. Derive it from the bulk modulus (compressibility = 1/B, in
        # A^3/eV) so those ensembles run out of the box and stay consistent with the barostat
        # stiffness used by `npt_nose_hoover`.
        if compressibility_au is None and ensemble.lower() in ("npt", "npt_berendsen"):
            if bulk_modulus <= 0:
                raise ValueError(
                    f"bulk_modulus must be positive to derive a compressibility, got {bulk_modulus}. "
                    "Pass compressibility_au explicitly instead."
                )
            compressibility_au = 1.0 / bulk_modulus

        if mask is None:
            mask = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        if external_stress is None:
            external_stress = 0.0

        if np.isclose(self.atoms.get_kinetic_energy(), 0.0, rtol=0, atol=1e-12):
            MaxwellBoltzmannDistribution(self.atoms, temperature_K=temperature)

        if ensemble.lower() == "nvt":
            self.dyn = NVTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                taut=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nve":
            self.dyn = VelocityVerlet(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_langevin":
            self.dyn = Langevin(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                friction=friction / units.fs,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_andersen":
            self.dyn = Andersen(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                andersen_prob=andersen_prob,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_bussi":
            self.dyn = Bussi(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                taut=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "nvt_nose_hoover_chain":
            self.dyn = NoseHooverChainNVT(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                tdamp=taut,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )
        elif ensemble.lower() == "npt":
            """
            NPT ensemble default to Inhomogeneous_NPTBerendsen thermo/barostat
            This is a more flexible scheme that fixes three angles of the unit
            cell but allows three lattice parameter to change independently.
            """

            self.dyn = Inhomogeneous_NPTBerendsen(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "npt_berendsen":
            """

            This is a similar scheme to the Inhomogeneous_NPTBerendsen.
            This is a less flexible scheme that fixes the shape of the
            cell - three angles are fixed and the ratios between the three
            lattice constants.

            """

            self.dyn = NPTBerendsen(
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility_au,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        elif ensemble.lower() == "npt_nose_hoover":
            self.upper_triangular_cell()
            self.dyn = NPT(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                externalstress=external_stress,  # type:ignore[arg-type]
                ttime=ttime * units.fs,
                # Already in ASE units (ptime**2 * B); see the pfactor note above.
                pfactor=pfactor,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
                mask=mask,
            )
        elif ensemble.lower() == "npt_nose_hoover_chain":
            self.dyn = IsotropicMTKNPT(  # type:ignore[assignment]
                self.atoms,
                timestep * units.fs,
                temperature_K=temperature,
                tdamp=taut,
                pdamp=taup,
                pressure_au=pressure,
                trajectory=trajectory,
                logfile=logfile,
                loginterval=loginterval,
                append_trajectory=append_trajectory,
            )

        else:
            raise ValueError("Ensemble not supported")

        self.trajectory = trajectory
        self.logfile = logfile
        self.loginterval = loginterval
        self.timestep = timestep
        # Kept so `set_atoms` can reseed velocities at the same temperature this run was
        # configured for, rather than leaving the new Atoms at rest.
        self.temperature = temperature

    def run(self, steps: int):
        """Thin wrapper of ase MD run.

        Args:
            steps (int): number of MD steps
        """
        self.dyn.run(steps)

    def set_atoms(self, atoms: Atoms):
        """Swap in a new Atoms object, reseeding velocities and restarting the step counter.

        Args:
            atoms (Atoms): new atoms for running MD.

        The new Atoms is a new simulation, not a continuation, so it needs its own velocities
        and its own step count. Previously this reassigned ``dyn.atoms`` and nothing else: the
        incoming Atoms kept whatever velocities it had (usually none), which made the measured
        temperature zero and the next ``run()`` die inside the thermostat with
        ``ZeroDivisionError: float division by zero`` as it scaled by ``T/T_old``. ``dyn.nsteps``
        also carried over, so trajectory frames and log lines continued numbering from the
        previous run.
        """
        if isinstance(atoms, Structure | Molecule):
            atoms = AseAtomsAdaptor().get_atoms(atoms)
        calculator = self.atoms.calc
        self.atoms = atoms
        self.dyn.atoms = atoms
        self.dyn.atoms.calc = calculator
        # Seed only if the caller did not supply velocities of their own, matching how the
        # constructor treats the Atoms it is handed.
        if np.isclose(atoms.get_kinetic_energy(), 0.0, rtol=0, atol=1e-12):
            MaxwellBoltzmannDistribution(atoms, temperature_K=self.temperature)
        # Restart the step counter so the new run's frames and log lines start from zero.
        self.dyn.nsteps = 0

    def upper_triangular_cell(self, verbose: bool | None = False) -> None:
        """Transform to upper-triangular cell.
        ASE Nose-Hoover implementation only supports upper-triangular cell
        while ASE's canonical description is lower-triangular cell.

        Args:
            verbose (bool): Whether to notify user about upper-triangular cell
                transformation. Default = False
        """
        if not NPT._isuppertriangular(self.atoms.get_cell()):
            a, b, c, alpha, beta, gamma = self.atoms.cell.cellpar()
            angles = np.radians((alpha, beta, gamma))
            sin_a, sin_b, _sin_g = np.sin(angles)
            cos_a, cos_b, cos_g = np.cos(angles)
            cos_p = (cos_g - cos_a * cos_b) / (sin_a * sin_b)
            cos_p = np.clip(cos_p, -1, 1)
            sin_p = (1 - cos_p**2) ** 0.5

            new_basis = [
                (a * sin_b * sin_p, a * sin_b * cos_p, a * cos_b),
                (0, b * sin_a, b * cos_a),
                (0, 0, c),
            ]

            # `set_cell(..., scale_atoms=True)` carries the *positions* into the new basis but
            # leaves `momenta` in the old frame, so velocities end up misoriented relative to
            # the lattice. Harmless for an isotropic Maxwell-Boltzmann seed (any orientation is
            # as good as another), but wrong whenever the incoming Atoms carries meaningful
            # velocities -- continuing a trajectory, or an anisotropically prepared velocity
            # field. Apply the same linear map to the momenta.
            #
            # For fractional coordinates f held fixed, r = f @ cell, so the map old -> new is
            # `inv(old_cell) @ new_cell` acting on row vectors. Recovered from the cells rather
            # than composed by hand so it stays exactly the transform ASE applied.
            old_cell = np.array(self.atoms.get_cell())
            momenta = self.atoms.get_momenta()
            self.atoms.set_cell(new_basis, scale_atoms=True)
            if np.abs(np.linalg.det(old_cell)) > 1e-12 and np.any(momenta):
                transform = np.linalg.solve(old_cell, np.array(self.atoms.get_cell()))
                self.atoms.set_momenta(momenta @ transform)
            if verbose:
                print("Transformed to upper triangular unit cell.", flush=True)
