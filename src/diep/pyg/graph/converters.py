"""Convert pymatgen structures and molecules into PyG graphs.

Produces exactly the same bond list, in the same order, as the DGL converters in
:mod:`diep.graph.converters` / :mod:`diep.ext.pymatgen` -- both call
``find_points_in_spheres`` and keep its ordering -- so a PyG graph and a DGL graph built
from the same structure are element-for-element comparable.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp
import torch
from pymatgen.core import Molecule, Structure
from pymatgen.core.periodic_table import Element
from pymatgen.optimization.neighbors import find_points_in_spheres

import diep
from diep.pyg.graph.compute import DIEPData

if TYPE_CHECKING:
    from collections.abc import Sequence


def get_element_list(train_structures: Sequence[Structure | Molecule]) -> tuple[str, ...]:
    """Get the tuple of elements in the training set for atomic features.

    Args:
        train_structures: pymatgen Molecule/Structure objects.

    Returns:
        Tuple of elements covered in the training set, ordered by atomic number.
    """
    elements: set[str] = set()
    for s in train_structures:
        elements.update(s.composition.get_el_amt_dict().keys())
    return tuple(sorted(elements, key=lambda el: Element(el).Z))


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to PyG graphs."""

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[DIEPData, torch.Tensor, np.ndarray | list]:
        """Convert a structure into a graph.

        Args:
            structure: Input crystal or molecule.

        Returns:
            (DIEPData, lattice, state_attr)
        """

    def get_graph_from_processed_structure(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        element_types,
        frac_coords,
        is_atoms: bool = False,
    ) -> tuple[DIEPData, torch.Tensor, np.ndarray]:
        """Construct a PyG graph from processed structure and bond information.

        Args:
            structure: Input crystal or molecule (or ASE atoms, see ``is_atoms``).
            src_id: site indices for the starting point of each bond.
            dst_id: site indices for the destination point of each bond.
            images: periodic image offsets for the bonds.
            lattice_matrix: lattice of the structure, shape (1, 3, 3).
            element_types: element symbols indexing the node types.
            frac_coords: fractional coordinates (Cartesian for molecules).
            is_atoms: whether the input structure is an ASE Atoms object.

        Returns:
            (DIEPData, lattice, state_attr)
        """
        device = torch.device("cpu")
        edge_index = torch.stack(
            [
                torch.as_tensor(np.asarray(src_id), dtype=torch.long, device=device),
                torch.as_tensor(np.asarray(dst_id), dtype=torch.long, device=device),
            ],
            dim=0,
        )
        lattice = torch.tensor(np.array(lattice_matrix), dtype=diep.float_th, device=device)
        element_to_index = {elem: idx for idx, elem in enumerate(element_types)}
        node_type = (
            np.array([element_types.index(site.specie.symbol) for site in structure])
            if is_atoms is False
            else np.array([element_to_index[elem] for elem in structure.get_chemical_symbols()])
        )
        data = DIEPData(
            edge_index=edge_index,
            num_nodes=len(structure),
            node_type=torch.tensor(node_type, dtype=diep.int_th, device=device),
            frac_coords=torch.tensor(np.asarray(frac_coords), dtype=diep.float_th, device=device),
            pbc_offset=torch.tensor(np.asarray(images), dtype=diep.float_th, device=device),
            lattice=lattice,
        )
        state_attr = np.array([0.0, 0.0]).astype(diep.float_np)
        return data, lattice, state_attr


class Structure2Graph(GraphConverter):
    """Construct a PyG graph from a pymatgen Structure."""

    def __init__(self, element_types: tuple[str, ...], cutoff: float = 5.0):
        """
        Args:
            element_types: elements present in the dataset, so every graph is built with the
                same node-type dimensionality.
            cutoff: cutoff radius for the graph representation.
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, structure: Structure) -> tuple[DIEPData, torch.Tensor, np.ndarray]:
        """Get a PyG graph from an input Structure.

        Args:
            structure: pymatgen Structure object.

        Returns:
            (DIEPData, lattice, state_attr)
        """
        numerical_tol = 1.0e-8
        pbc = np.array([1, 1, 1], dtype=np.int64)
        lattice_matrix = structure.lattice.matrix
        cart_coords = structure.cart_coords
        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            cart_coords, cart_coords, r=self.cutoff, pbc=pbc, lattice=lattice_matrix, tol=numerical_tol
        )
        exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
        src_id, dst_id, images = src_id[exclude_self], dst_id[exclude_self], images[exclude_self]
        return self.get_graph_from_processed_structure(
            structure, src_id, dst_id, images, [lattice_matrix], self.element_types, structure.frac_coords
        )


class Molecule2Graph(GraphConverter):
    """Construct a PyG graph from a pymatgen Molecule."""

    def __init__(self, element_types: tuple[str, ...], cutoff: float = 5.0):
        """
        Args:
            element_types: elements present in the dataset.
            cutoff: cutoff radius for the graph representation.
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, mol: Molecule) -> tuple[DIEPData, torch.Tensor, list]:
        """Get a PyG graph from an input Molecule.

        Args:
            mol: pymatgen Molecule object.

        Returns:
            (DIEPData, identity lattice, state_attr)
        """
        natoms = len(mol)
        R = mol.cart_coords
        weight = mol.composition.weight / len(mol)
        dist = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
        dists = mol.distance_matrix.flatten()
        nbonds = (np.count_nonzero(dists <= self.cutoff) - natoms) / 2
        nbonds /= natoms
        adj = sp.csr_matrix(dist <= self.cutoff) - sp.eye(natoms, dtype=np.bool_)
        adj = adj.tocoo()
        data, lat, _ = self.get_graph_from_processed_structure(
            mol,
            adj.row,
            adj.col,
            np.zeros((len(adj.row), 3)),
            np.expand_dims(np.identity(3), axis=0),
            self.element_types,
            R,
        )
        return data, lat, [weight, nbonds]
