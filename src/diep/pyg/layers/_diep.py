"""DIEP electron-ion potential integration on PyG graphs.

The physics -- grid, Gaussian density, local 2D frames, triplet canonicalisation -- is
shared verbatim with the DGL implementation in :mod:`diep.layers._diep`, which is pure
torch and framework-agnostic. Only the graph plumbing is rewritten here.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

import diep
from diep.layers._diep import (
    DIEPGrid,
    _build_bond_frames_vectorized,
    _canonicalize_triplets_batch,
    _gaussian_density,
    _project_points_batch,
)


class DIEPIntegrator(nn.Module):
    """Compute DIEP bond and triplet embeddings via electron-ion potential integration.

    Parameter-free; kept as a Module so it participates in ``.to(device)`` and mirrors the
    DGL implementation's place in the model tree.
    """

    def __init__(
        self,
        grid_half_length: float = 5.0,
        base_spacing: float = 1.0,
        sigma: float = 1.0,
        mode: Literal["sum", "grid"] = "sum",
        softening_epsilon: float = 0.5,
        use_effective_charge: bool = True,
    ):
        """
        Args:
            grid_half_length: half-length of the 2D integration grid.
            base_spacing: grid spacing.
            sigma: width parameter for the Gaussian electron density.
            mode: "sum" for a scalar per bond, "grid" for the full grid features.
            softening_epsilon: softening parameter preventing 1/r singularities.
            use_effective_charge: use sqrt(Z) instead of Z.
        """
        super().__init__()
        if sigma <= 0:
            raise ValueError("Sigma must be positive.")
        if base_spacing <= 0:
            raise ValueError("Grid spacing must be positive.")
        if grid_half_length <= 0:
            raise ValueError("Grid half length must be positive.")
        if softening_epsilon < 0:
            raise ValueError("Softening epsilon must be non-negative.")

        self.grid_half_length = grid_half_length
        self.base_spacing = base_spacing
        self.sigma = sigma
        self.mode = mode
        self.softening_epsilon = softening_epsilon
        self.use_effective_charge = use_effective_charge
        self._grid: DIEPGrid | None = None
        self._grid_device: torch.device | None = None

    @property
    def edge_dim(self) -> int:
        """Width of the produced bond / triplet feature vector."""
        if self.mode == "grid":
            num_axis = int(round(2 * self.grid_half_length / self.base_spacing)) + 1
            return num_axis * num_axis
        return 1

    def _ensure_grid(self, device: torch.device):
        if self._grid_device == device and self._grid is not None:
            return
        self._grid = DIEPGrid(self.grid_half_length, self.base_spacing, device)
        self._grid_device = device

    def _empty(self, device):
        width = 1 if self.mode == "sum" else self.edge_dim
        empty = torch.zeros((0, width), dtype=diep.float_th, device=device)
        return empty, empty.clone()

    def forward(self, data, atomic_numbers: torch.Tensor, compute_triplets: bool = True):
        """Return DIEP bond and triplet features.

        Args:
            data: graph carrying ``edge_index``, ``pos``, ``pbc_offshift`` and -- when
                ``compute_triplets`` -- ``triple_index`` and ``bond_vec``.
            atomic_numbers: atomic number of each node.
            compute_triplets: whether to compute the triplet features.

        Returns:
            (bond_feat, triplet_feat)
        """
        device = data.edge_index.device
        self._ensure_grid(device)
        if self._grid is None:
            raise RuntimeError("Integration grid was not initialised.")

        src, dst = data.edge_index
        n_edges = int(src.shape[0])
        if n_edges == 0:
            return self._empty(device)

        z_src = atomic_numbers[src].to(diep.float_th)
        z_dst = atomic_numbers[dst].to(diep.float_th)
        pos = data.pos.to(diep.float_th)

        pbc_offshift = getattr(data, "pbc_offshift", None)
        if pbc_offshift is None:
            pbc_offshift = torch.zeros((n_edges, 3), dtype=diep.float_th, device=device)
        else:
            pbc_offshift = pbc_offshift.to(diep.float_th)

        grid = self._grid
        grid_points = grid.points.to(diep.float_th)  # (P, 2)
        delta_area = grid.delta_area.to(diep.float_th)

        pos_src_batch = pos[src]
        pos_dst_batch = pos[dst] + pbc_offshift

        origins, e_x_batch, e_y_batch = _build_bond_frames_vectorized(pos_src_batch, pos_dst_batch)
        atom_positions_3d = torch.stack([pos_src_batch, pos_dst_batch], dim=1)  # (E, 2, 3)
        fragment_coords = _project_points_batch(atom_positions_3d, origins, e_x_batch, e_y_batch)

        diff_fragment = fragment_coords.unsqueeze(2) - grid_points.unsqueeze(0).unsqueeze(0)
        dist_sq_fragment = torch.sum(diff_fragment**2, dim=-1)  # (E, 2, P)
        rho_total = _gaussian_density(dist_sq_fragment, self.sigma).sum(dim=1)  # (E, P)

        eps_sq = self.softening_epsilon**2
        denom = torch.sqrt(dist_sq_fragment + eps_sq)
        if self.use_effective_charge:
            z_eff = torch.stack([torch.sqrt(z_src), torch.sqrt(z_dst)], dim=1)
        else:
            z_eff = torch.stack([z_src, z_dst], dim=1)
        potential = (z_eff.unsqueeze(2) / denom).sum(dim=1)  # (E, P)
        integrand = delta_area * rho_total * potential
        bond_feat = integrand.sum(dim=1, keepdim=True) if self.mode == "sum" else integrand

        if not compute_triplets:
            _, empty = self._empty(device)
            return bond_feat.to(diep.float_th), empty

        triple_index = getattr(data, "triple_index", None)
        if triple_index is None:
            raise ValueError("triple_index must be present when compute_triplets is True.")

        # Index-space precondition: triple_index holds *parent-bond* ids, so it indexes
        # src / dst / bond_vec (all length data.num_edges) directly. create_line_graph
        # guarantees this; assert_lg_invariants checks it. With pruned-bond ids here the
        # triplet geometry itself would be built from the wrong bonds.
        lg_src, lg_dst = triple_index
        if int(lg_src.shape[0]) == 0:
            _, empty = self._empty(device)
            return bond_feat.to(diep.float_th), empty

        bond_vec = data.bond_vec.to(diep.float_th)
        center_idx = src[lg_src]
        neighbor_i = dst[lg_src]
        neighbor_k = dst[lg_dst]

        pos_center = pos[center_idx]
        pos_first = pos_center + bond_vec[lg_src]
        pos_second = pos_center + bond_vec[lg_dst]

        raw_coords = torch.stack([pos_first, pos_center, pos_second], dim=1)
        triplet_indices = torch.stack([neighbor_i, center_idx, neighbor_k], dim=1)
        triplet_numbers = atomic_numbers[triplet_indices]

        canonical_coords, ordered_numbers = _canonicalize_triplets_batch(raw_coords, triplet_numbers)

        diff_fragment = canonical_coords.unsqueeze(2) - grid_points.unsqueeze(0)
        dist_sq_fragment = torch.sum(diff_fragment**2, dim=-1)
        rho_total = _gaussian_density(dist_sq_fragment, self.sigma).sum(dim=1)

        denom_triplet = torch.sqrt(dist_sq_fragment + eps_sq)
        if self.use_effective_charge:
            ordered_numbers_eff = torch.sqrt(ordered_numbers.clamp_min(0.0))
        else:
            ordered_numbers_eff = ordered_numbers
        potential_triplet = (ordered_numbers_eff.unsqueeze(-1) / denom_triplet).sum(dim=1)
        integrand = delta_area * rho_total * potential_triplet
        triplet_feat = integrand.sum(dim=1, keepdim=True) if self.mode == "sum" else integrand

        return bond_feat.to(diep.float_th), triplet_feat.to(diep.float_th)
