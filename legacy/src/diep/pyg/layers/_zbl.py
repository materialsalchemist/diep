"""ZBL repulsive potential for the PyG backend.

Translation of :mod:`diep.layers._zbl`; the original follows the SchNetPack implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from ase.data import atomic_numbers
from torch import nn
from torch_geometric.utils import scatter

import diep
from diep.layers._activations import softplus_inverse
from diep.utils.cutoff import polynomial_cutoff


class NuclearRepulsion(nn.Module):
    """Ziegler-Biersack-Littmark style repulsion energy."""

    def __init__(
        self,
        r_cut: float,
        a0: float = 0.5291772105638411,
        ke: float = 14.399645351950548,
        trainable: bool = False,
    ):
        """
        Args:
            r_cut: cutoff for the interaction range.
            a0: distance unit conversion from bohr into angstrom.
            ke: coulomb constant unit conversion from Ha into eV.
            trainable: whether the parameters are trainable.
        """
        super().__init__()
        self.r_cut = r_cut
        self.register_buffer("ke", torch.tensor(ke, dtype=diep.float_th))

        a_div = softplus_inverse(torch.tensor([1.0 / (a0 * 0.8854)], dtype=diep.float_th))
        a_pow = softplus_inverse(torch.tensor([0.23], dtype=diep.float_th))
        exponents = softplus_inverse(torch.tensor([3.19980, 0.94229, 0.40290, 0.20162], dtype=diep.float_th))
        coefficients = softplus_inverse(torch.tensor([0.18175, 0.50986, 0.28022, 0.02817], dtype=diep.float_th))

        self.a_pow = nn.Parameter(a_pow, requires_grad=trainable)
        self.a_div = nn.Parameter(a_div, requires_grad=trainable)
        self.coefficients = nn.Parameter(coefficients, requires_grad=trainable)
        self.exponents = nn.Parameter(exponents, requires_grad=trainable)

    def forward(self, element_types: tuple, data):
        """Pairwise ZBL nuclear repulsive energy.

        Args:
            element_types: element symbols indexing the node types.
            data: PyG graph carrying ``node_type``, ``edge_index`` and ``bond_dist``.

        Returns:
            One repulsion energy per structure.
        """
        device = data.edge_index.device
        z_list = torch.tensor([atomic_numbers[i] for i in element_types], dtype=diep.float_th, device=device)
        z = z_list[data.node_type.long()]

        idx_i, idx_j = data.edge_index
        r_ij = data.bond_dist

        a = z ** F.softplus(self.a_pow)
        a_ij = (a[idx_i] + a[idx_j]) * F.softplus(self.a_div)
        exponents = a_ij[..., None] * F.softplus(self.exponents)[None, ...]
        coefficients = F.normalize(F.softplus(self.coefficients)[None, ...], p=1.0, dim=1)
        screening = torch.sum(coefficients * torch.exp(-exponents * r_ij[:, None]), dim=1)

        eij = (z[idx_i] * z[idx_j]) * polynomial_cutoff(r_ij, self.r_cut) * screening / r_ij

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(z.numel(), dtype=torch.long, device=device)
        size = int(batch.max()) + 1 if batch.numel() else 1
        # DGL: update_all(copy_e, sum) onto destination nodes, then readout_nodes(sum)
        per_node = scatter(eij, idx_j.long(), dim=0, dim_size=z.numel(), reduce="sum")
        energy = 0.5 * self.ke * scatter(per_node, batch, dim=0, dim_size=size, reduce="sum")
        return torch.squeeze(energy)
