"""Interatomic potential built on the PyG DIEP model.

Translation of :mod:`diep.apps.pes`. Differences from the DGL version, both deliberate:

* ``use_edges`` defaults to ``None`` and is only applied when it is not None. The DGL
  version defaults it to ``False`` and guards with ``if use_edges is not None:``, which is
  true for ``False`` -- so constructing a DGL ``Potential`` always switches the triplet path
  off, including in ``PotentialLightningModule``. Here, leaving it unset keeps whatever the
  model was built with, and passing ``use_edges=False`` still switches it off explicitly.
* Chunked evaluation is supported through ``num_chunks`` / ``apply_chunking`` as on the DGL
  side, using PyG subgraph extraction.
"""

from __future__ import annotations

import os
import secrets

import numpy as np
import torch
from torch import nn
from torch.autograd import grad

import diep
from diep.pyg.layers import AtomRef, NuclearRepulsion
from diep.utils.io import IOMixIn


def _node_subgraph(data, mask: torch.Tensor):
    """Extract the induced subgraph on the masked nodes, keeping node/edge attributes.

    Returns the subgraph and the original node ids of its nodes (the analogue of
    ``dgl.node_subgraph`` plus ``subg.ndata[dgl.NID]``).
    """
    device = data.edge_index.device
    node_ids = mask.nonzero(as_tuple=False).reshape(-1)
    remap = torch.full((int(data.num_nodes),), -1, dtype=torch.long, device=device)
    remap[node_ids] = torch.arange(node_ids.numel(), dtype=torch.long, device=device)

    src, dst = data.edge_index
    keep = mask[src] & mask[dst]
    sub = data.__class__()
    sub.edge_index = torch.stack([remap[src[keep]], remap[dst[keep]]], dim=0)
    sub.num_nodes = int(node_ids.numel())
    for key in ("node_type", "frac_coords", "pos"):
        value = getattr(data, key, None)
        if value is not None:
            setattr(sub, key, value[node_ids])
    for key in ("pbc_offset", "pbc_offshift", "bond_vec", "bond_dist"):
        value = getattr(data, key, None)
        if value is not None:
            setattr(sub, key, value[keep])
    lattice = getattr(data, "lattice", None)
    if lattice is not None:
        sub.lattice = lattice
    return sub, node_ids


class Potential(nn.Module, IOMixIn):
    """A potential for energies, forces, stresses and hessians."""

    __version__ = 1

    def __init__(
        self,
        model: nn.Module,
        data_mean: torch.Tensor | float = 0.0,
        data_std: torch.Tensor | float = 1.0,
        element_refs: torch.Tensor | np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
        calc_magmom: bool = False,
        calc_repuls: bool = False,
        zbl_trainable: bool = False,
        debug_mode: bool = False,
        use_edges: bool | None = None,
        num_chunks: int | None = None,
        chunk_padding: float | None = None,
        apply_chunking: bool = False,
        write_chunk_logs: bool = True,
    ):
        """
        Args:
            model: model predicting energies.
            data_mean: mean of the training target.
            data_std: standard deviation of the training target.
            element_refs: element reference values for each element.
            calc_forces: enable force calculations.
            calc_stresses: enable stress calculations.
            calc_hessian: enable hessian calculations.
            calc_magmom: enable site-wise property calculation.
            calc_repuls: whether ZBL repulsion is included.
            zbl_trainable: whether the ZBL repulsion is trainable.
            debug_mode: return raw gradients for checking.
            use_edges: if not None, overrides the model's triplet/line-graph usage. Leaving
                it None keeps whatever the model was constructed with.
            num_chunks: None or 0 disables chunking. Otherwise split each axis into
                (num_chunks + 1) parts.
            chunk_padding: padding around each chunk (at least the model cutoff).
            apply_chunking: compute energies/forces from chunks only.
            write_chunk_logs: whether to write chunk XYZ files.
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian
        self.calc_magmom = calc_magmom
        self.debug_mode = debug_mode
        self.calc_repuls = calc_repuls
        self.use_edges = use_edges
        self.num_chunks = num_chunks
        self.chunk_padding = chunk_padding
        self.apply_chunking = apply_chunking
        self.write_chunk_logs = write_chunk_logs

        if use_edges is not None:
            if hasattr(self.model, "use_edges"):
                self.model.use_edges = use_edges
            if hasattr(self.model, "use_triplets"):
                self.model.use_triplets = use_edges

        if calc_repuls:
            self.repuls = NuclearRepulsion(self.model.cutoff, trainable=zbl_trainable)

        self.element_refs: AtomRef | None
        if element_refs is not None:
            if not isinstance(element_refs, torch.Tensor):
                element_refs = torch.tensor(element_refs, dtype=diep.float_th)
            self.element_refs = AtomRef(property_offset=element_refs)
        else:
            self.element_refs = None

        if data_mean is None:
            data_mean = 0.0
        if not isinstance(data_mean, torch.Tensor):
            data_mean = torch.tensor(data_mean, dtype=diep.float_th)
        if not isinstance(data_std, torch.Tensor):
            data_std = torch.tensor(data_std, dtype=diep.float_th)
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_std", data_std)

    @staticmethod
    def _batch_size(g) -> int:
        batch = getattr(g, "batch", None)
        return 1 if batch is None or batch.numel() == 0 else int(batch.max()) + 1

    @staticmethod
    def _num_nodes_per_graph(g) -> torch.Tensor:
        batch = getattr(g, "batch", None)
        if batch is None:
            return torch.tensor([int(g.num_nodes)], device=g.edge_index.device)
        return torch.bincount(batch, minlength=int(batch.max()) + 1)

    @staticmethod
    def _num_edges_per_graph(g) -> torch.Tensor:
        batch = getattr(g, "batch", None)
        src = g.edge_index[0]
        if batch is None:
            return torch.tensor([int(src.numel())], device=src.device)
        return torch.bincount(batch[src], minlength=int(batch.max()) + 1)

    def _set_positions(self, g, lattice: torch.Tensor):
        """Recompute per-edge offshifts and per-atom positions from the (strained) lattice."""
        edge_lattice = torch.repeat_interleave(lattice, self._num_edges_per_graph(g), dim=0)
        g.lattice_per_edge = edge_lattice
        g.pbc_offshift = (g.pbc_offset.unsqueeze(dim=-1) * edge_lattice).sum(dim=1)
        node_lattice = torch.repeat_interleave(lattice, self._num_nodes_per_graph(g), dim=0)
        g.pos = (g.frac_coords.unsqueeze(dim=-1) * node_lattice).sum(dim=1)

    def _chunk_masks(self, frac_coords, center, half_width, pad_frac):
        delta = frac_coords - center
        delta = delta - torch.round(delta)
        abs_delta = delta.abs()
        return (abs_delta < half_width).all(dim=1), (abs_delta < (half_width + pad_frac)).all(dim=1)

    def _compute_energy_forces(self, g, lattice: torch.Tensor, state_attr: torch.Tensor | None):
        self._set_positions(g, lattice)
        g.pos.requires_grad_(True)
        total_energies = self.model(g=g, state_attr=state_attr)
        total_energies = self.data_std * total_energies + self.data_mean
        if self.calc_repuls:
            total_energies = total_energies + self.repuls(self.model.element_types, g)
        if self.element_refs is not None:
            total_energies = total_energies + torch.squeeze(self.element_refs(g))
        grads = grad(
            total_energies,
            [g.pos],
            grad_outputs=torch.ones_like(total_energies),
            create_graph=False,
            retain_graph=False,
        )
        return total_energies, -grads[0]

    def _chunk_padding_frac(self, lattice: torch.Tensor) -> tuple[float, float, float]:
        lengths = [float(torch.linalg.norm(lattice[0][i])) for i in range(3)]
        if self.chunk_padding is None:
            nblocks = int(getattr(self.model, "n_blocks", 1))
            pad = float(self.model.cutoff) * nblocks
            if self.use_edges is not False:
                pad = max(pad, float(getattr(self.model, "threebody_cutoff", 0.0)))
        else:
            pad = self.chunk_padding
        pad = max(pad, float(self.model.cutoff))
        return tuple(pad / length if length > 0 else 0.0 for length in lengths)

    def _write_chunk_xyz(self, path, symbols, positions, forces_chunk, forces_full, energy):
        positions = positions.detach().cpu().numpy()
        forces_chunk = forces_chunk.detach().cpu().numpy()
        forces_full = forces_full.detach().cpu().numpy()
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(f"{len(symbols)}\n")
            handle.write(f"energy={energy.item():.8f}\n")
            for sym, pos, fc, ff in zip(symbols, positions, forces_chunk, forces_full, strict=False):
                handle.write(
                    f"{sym} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f} "
                    f"{fc[0]:.8f} {fc[1]:.8f} {fc[2]:.8f} {ff[0]:.8f} {ff[1]:.8f} {ff[2]:.8f}\n"
                )

    def _chunked_energy_forces(self, g, lattice, state_attr):
        """Sum chunk energies and gather chunk forces over a partition of the cell."""
        if self._batch_size(g) != 1:
            raise RuntimeError("Chunking only supports a single structure (batch_size=1).")
        pad_frac = torch.tensor(self._chunk_padding_frac(lattice), device=g.frac_coords.device,
                                dtype=g.frac_coords.dtype)
        frac_coords = g.frac_coords - torch.floor(g.frac_coords)
        divisions = self.num_chunks + 1
        chunk_width = 1.0 / float(divisions)
        half_width = chunk_width * 0.5

        run_id = None
        if self.write_chunk_logs:
            os.makedirs("chunking_log", exist_ok=True)
            run_id = os.path.join("chunking_log", "chunks_" + secrets.token_hex(4))
            os.makedirs(run_id, exist_ok=True)

        full_forces = torch.zeros_like(g.frac_coords)
        total_energy = None
        n_chunk = 0
        for ix in range(divisions):
            for iy in range(divisions):
                for iz in range(divisions):
                    center = frac_coords.new_tensor(
                        [ix * chunk_width + half_width, iy * chunk_width + half_width,
                         iz * chunk_width + half_width]
                    )
                    core_mask, pad_mask = self._chunk_masks(frac_coords, center, half_width, pad_frac)
                    if not torch.any(core_mask):
                        continue
                    subg, orig_ids = _node_subgraph(g, pad_mask)
                    core_subg_mask = core_mask[orig_ids]
                    if not torch.any(core_subg_mask):
                        continue
                    chunk_energy, chunk_forces = self._compute_energy_forces(subg, lattice, state_attr)
                    full_forces[orig_ids[core_subg_mask]] = chunk_forces[core_subg_mask]
                    total_energy = chunk_energy if total_energy is None else total_energy + chunk_energy
                    if run_id is not None:
                        core_frac = subg.frac_coords[core_subg_mask]
                        core_frac = core_frac - torch.floor(core_frac)
                        symbols = [
                            self.model.element_types[int(i)]
                            for i in subg.node_type[core_subg_mask].detach().cpu().tolist()
                        ]
                        self._write_chunk_xyz(
                            os.path.join(run_id, f"chunk_{n_chunk:04d}.xyz"),
                            symbols,
                            core_frac @ lattice[0],
                            chunk_forces[core_subg_mask],
                            full_forces[orig_ids[core_subg_mask]],
                            torch.atleast_1d(chunk_energy)[0],
                        )
                    n_chunk += 1
        if total_energy is None:
            total_energy = torch.zeros(1, device=lattice.device)
        return total_energy, full_forces

    def forward(self, g, lat: torch.Tensor, state_attr: torch.Tensor | None = None, l_g=None):
        """Predict energies and their derivatives.

        Args:
            g: PyG graph or batch.
            lat: lattice matrices, shape (B, 3, 3).
            state_attr: state attributes.
            l_g: ignored; the line graph travels on ``g``. Accepted for API parity.

        Returns:
            (energies, forces, stresses, hessian) or, with ``calc_magmom``,
            (energies, forces, stresses, hessian, site-wise properties).
        """
        del l_g
        batch_size = self._batch_size(g)
        lat = torch.atleast_3d(lat) if lat.dim() < 3 else lat
        st = lat.new_zeros([batch_size, 3, 3])
        if self.calc_stresses:
            st.requires_grad_(True)
        lattice = lat @ (torch.eye(3, device=lat.device) + st)

        if isinstance(state_attr, torch.Tensor) and state_attr.device != lat.device:
            state_attr = state_attr.to(lat.device)

        if self.apply_chunking:
            if not self.num_chunks:
                self.apply_chunking = False
                return self.forward(g=g, lat=lat, state_attr=state_attr)
            total_energy, full_forces = self._chunked_energy_forces(g, lattice, state_attr)
            return total_energy, full_forces, torch.zeros(1), torch.zeros(1)

        self._set_positions(g, lattice)
        if self.calc_forces:
            g.pos.requires_grad_(True)

        if self.use_edges is False:
            if hasattr(self.model, "use_edges"):
                self.model.use_edges = False
        total_energies = self.model(g=g, state_attr=state_attr)
        total_energies = self.data_std * total_energies + self.data_mean

        if self.calc_repuls:
            total_energies = total_energies + self.repuls(self.model.element_types, g)
        if self.element_refs is not None:
            total_energies = total_energies + torch.squeeze(self.element_refs(g))

        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)
        grads = None

        grad_vars = [g.pos, st] if self.calc_stresses else [g.pos]
        if self.calc_forces:
            grads = grad(
                total_energies,
                grad_vars,
                grad_outputs=torch.ones_like(total_energies),
                create_graph=True,
                retain_graph=True,
            )
            forces = -grads[0]

        if self.calc_hessian:
            r = grads[0].view(-1)
            n = r.size(0)
            hessian = total_energies.new_zeros((n, n))
            for iatom in range(n):
                tmp = grad([r[iatom]], g.pos, retain_graph=iatom < n)[0]
                if tmp is not None:
                    hessian[iatom] = tmp.view(-1)

        if self.calc_stresses:
            volume = torch.abs(torch.det(lattice))
            sts = grads[1]
            scale = 1.0 / volume * 160.21766208
            sts = [i * j for i, j in zip(sts, scale, strict=False)] if sts.dim() == 3 else [sts * scale]
            stresses = torch.cat(sts)

        if self.debug_mode:
            return total_energies, grads[0], grads[1]
        if self.calc_magmom:
            return total_energies, forces, stresses, hessian, g.magmom
        return total_energies, forces, stresses, hessian
