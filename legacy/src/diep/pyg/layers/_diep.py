"""DIEP electron-ion potential integration on PyG graphs.

The physics -- grid, Gaussian density, local 2D frames, triplet canonicalisation -- is
shared verbatim with the DGL implementation in :mod:`diep.layers._diep`, which is pure
torch and framework-agnostic. Only the graph plumbing is rewritten here.
"""

from __future__ import annotations

from collections.abc import Sequence
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

    In "sum" mode the physics integral itself still collapses each bond/triplet to a single
    scalar (as it always has) -- but that scalar can then be gated by a bank of Gaussian
    windows in bond-length space, one channel per window center, so each channel only
    responds to bonds near its own center. That gating is what actually decorrelates the
    channels: varying the electron-density width alone keeps every channel a smooth,
    monotonic function of the same underlying bond length, which stays highly correlated
    across the observed distance range; windows centered at different lengths instead give
    channels with mostly disjoint support.

    Kept as a Module so it participates in ``.to(device)`` and mirrors the DGL
    implementation's place in the model tree.
    """

    def __init__(
        self,
        grid_half_length: float = 5.0,
        base_spacing: float = 1.0,
        sigma: float = 1.0,
        mode: Literal["sum", "grid"] = "sum",
        softening_epsilon: float = 0.5,
        use_effective_charge: bool = True,
        channel_centers: float | Sequence[float] | None = None,
        channel_width: float | None = None,
    ):
        """
        Args:
            grid_half_length: half-length of the 2D integration grid.
            base_spacing: grid spacing.
            sigma: width parameter for the Gaussian electron density.
            mode: "sum" for a vector per bond, "grid" for the full grid features. The
                bond-length channel gating below only applies in "sum" mode.
            softening_epsilon: softening parameter preventing 1/r singularities.
            use_effective_charge: use sqrt(Z) instead of Z.
            channel_centers: bond-length centers (Angstrom) of a bank of Gaussian windows
                gating the physics-integral scalar -- one channel per center. ``None``
                (default) disables gating: a single, ungated channel, matching the original
                behaviour.
            channel_width: width (Angstrom) of each Gaussian window. Required, and must be
                positive, whenever ``channel_centers`` gives more than one value.
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

        centers_list = None
        if channel_centers is not None:
            centers_list = (
                [float(channel_centers)]
                if isinstance(channel_centers, (int, float))
                else [float(c) for c in channel_centers]
            )
            if not centers_list:
                raise ValueError("channel_centers, if given, must be non-empty.")
            if mode == "grid" and len(centers_list) > 1:
                raise ValueError("Multiple channel centers are only supported in mode='sum'.")
            if len(centers_list) > 1 and not (channel_width and channel_width > 0):
                raise ValueError("channel_width must be a positive float when more than one center is given.")

        self.grid_half_length = grid_half_length
        self.base_spacing = base_spacing
        self.sigma = sigma
        self.mode = mode
        self.softening_epsilon = softening_epsilon
        self.use_effective_charge = use_effective_charge
        self.channel_width = channel_width
        self.register_buffer(
            "channel_centers",
            None if centers_list is None else torch.tensor(centers_list, dtype=diep.float_th),
            persistent=False,
        )
        self._grid: DIEPGrid | None = None
        self._grid_device: torch.device | None = None

    @property
    def edge_dim(self) -> int:
        """Width of the produced bond / triplet feature vector."""
        if self.mode == "grid":
            num_axis = int(round(2 * self.grid_half_length / self.base_spacing)) + 1
            return num_axis * num_axis
        return 1 if self.channel_centers is None else int(self.channel_centers.numel())

    def _ensure_grid(self, device: torch.device):
        if self._grid_device == device and self._grid is not None:
            return
        self._grid = DIEPGrid(self.grid_half_length, self.base_spacing, device)
        self._grid_device = device

    def _empty(self, device):
        empty = torch.zeros((0, self.edge_dim), dtype=diep.float_th, device=device)
        return empty, empty.clone()

    def _integrate(
        self,
        dist_sq_fragment: torch.Tensor,
        denom: torch.Tensor,
        z_eff: torch.Tensor,
        delta_area: torch.Tensor,
        length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate density x potential over the grid, then (in "sum" mode) gate the result.

        Args:
            dist_sq_fragment: (N, k, P) squared distances from the k source atoms to each
                grid point.
            denom: softened distance, ``sqrt(dist_sq_fragment + softening_epsilon**2)``.
            z_eff: (N, k) (effective) charge of each of the k source atoms.
            delta_area: grid cell area.
            length: (N,) characteristic bond length used to gate each channel; required
                whenever more than one channel center is configured.

        Returns:
            (N, edge_dim): grid mode returns one value per grid point; sum mode returns one
            value per channel (a single ungated channel when ``channel_centers`` is None).
        """
        potential = (z_eff.unsqueeze(-1) / denom).sum(dim=1)  # (N, P)
        rho_total = _gaussian_density(dist_sq_fragment, self.sigma).sum(dim=1)  # (N, P)
        integrand = delta_area * rho_total * potential  # (N, P)
        if self.mode == "grid":
            return integrand
        base = integrand.sum(dim=1, keepdim=True)  # (N, 1)
        if self.channel_centers is None:
            return base
        window = torch.exp(-0.5 * ((length.unsqueeze(-1) - self.channel_centers) / self.channel_width) ** 2)
        return base * window  # (N, D)

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

        eps_sq = self.softening_epsilon**2
        denom = torch.sqrt(dist_sq_fragment + eps_sq)
        if self.use_effective_charge:
            z_eff = torch.stack([torch.sqrt(z_src), torch.sqrt(z_dst)], dim=1)
        else:
            z_eff = torch.stack([z_src, z_dst], dim=1)
        bond_length = getattr(data, "bond_dist", None)
        if bond_length is None:
            bond_length = torch.norm(pos_dst_batch - pos_src_batch, dim=-1)
        bond_feat = self._integrate(dist_sq_fragment, denom, z_eff, delta_area, length=bond_length.to(diep.float_th))

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

        denom_triplet = torch.sqrt(dist_sq_fragment + eps_sq)
        if self.use_effective_charge:
            ordered_numbers_eff = torch.sqrt(ordered_numbers.clamp_min(0.0))
        else:
            ordered_numbers_eff = ordered_numbers
        bond_dist = data.bond_dist.to(diep.float_th)
        triplet_length = 0.5 * (bond_dist[lg_src] + bond_dist[lg_dst])
        triplet_feat = self._integrate(
            dist_sq_fragment, denom_triplet, ordered_numbers_eff, delta_area, length=triplet_length
        )

        return bond_feat.to(diep.float_th), triplet_feat.to(diep.float_th)
