"""Implementation of Interatomic Potentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.autograd import grad

import diep
from diep.layers import AtomRef, NuclearRepulsion
from diep.utils.io import IOMixIn

if TYPE_CHECKING:
    import dgl
    import numpy as np


class Potential(nn.Module, IOMixIn):
    """A class representing an interatomic potential."""

    __version__ = 2

    def __init__(
        self,
        model: nn.Module,
        data_mean: torch.Tensor | float = 0.0,
        data_std: torch.Tensor | float = 1.0,
        element_refs: np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
        calc_site_wise: bool = False,
        calc_repuls: bool = False,
        zbl_trainable: bool = False,
        debug_mode: bool = False,
    ):
        """Initialize Potential from a model and elemental references.

        Args:
            model: Model for predicting energies.
            data_mean: Mean of target.
            data_std: Std dev of target.
            element_refs: Element reference values for each element.
            calc_forces: Enable force calculations.
            calc_stresses: Enable stress calculations.
            calc_hessian: Enable hessian calculations.
            calc_site_wise: Enable site-wise property calculation.
            calc_repuls: Whether the ZBL repulsion is included
            zbl_trainable: Whether zbl repulsion is trainable
            debug_mode: Return gradient of total energy with respect to atomic positions and lattices for checking
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian
        self.calc_site_wise = calc_site_wise
        self.element_refs: AtomRef | None
        self.debug_mode = debug_mode
        self.calc_repuls = calc_repuls

        if calc_repuls:
            self.repuls = NuclearRepulsion(self.model.cutoff, trainable=zbl_trainable)

        if element_refs is not None:
            self.element_refs = AtomRef(property_offset=torch.tensor(element_refs, dtype=diep.float_th))
        else:
            self.element_refs = None
        # for backward compatibility
        if data_mean is None:
            data_mean = 0.0
        self.register_buffer("data_mean", torch.tensor(data_mean, dtype=diep.float_th))
        self.register_buffer("data_std", torch.tensor(data_std, dtype=diep.float_th))

    def forward(
        self,
        g: dgl.DGLGraph,
        lat: torch.Tensor,
        state_attr: torch.Tensor | None = None,
        l_g: dgl.DGLGraph | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Args:
            g: DGL graph
            lat: lattice
            state_attr: State attrs
            l_g: Line graph.

        Returns:
            (energies, forces, stresses, hessian) or (energies, forces, stresses, hessian, site-wise properties)
        """
        # st (strain) for stress calculations
        st = lat.new_zeros([g.batch_size, 3, 3])
        if self.calc_stresses:
            st.requires_grad_(True)
        lattice = lat @ (torch.eye(3).to(st.device) + st)
        g.edata["lattice"] = torch.repeat_interleave(lattice, g.batch_num_edges(), dim=0)
        g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)

        predictions = self.model(g=g, state_attr=state_attr, l_g=l_g)
        if isinstance(predictions, tuple) and len(predictions) > 1:
            total_energies, site_wise = predictions
        else:
            total_energies = predictions
            site_wise = None

        total_energies = self.data_std * total_energies + self.data_mean

        if self.calc_repuls:
            total_energies += self.repuls(self.model.element_types, g)

        if self.element_refs is not None:
            property_offset = torch.squeeze(self.element_refs(g))
            total_energies += property_offset

        forces = torch.zeros(1)
        stresses = torch.zeros(1)
        hessian = torch.zeros(1)

        grad_vars = [g.ndata["pos"], st] if self.calc_stresses else [g.ndata["pos"]]        
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
            r = -grads[0].view(-1)
            s = r.size(0)
            hessian = total_energies.new_zeros((s, s))
            for iatom in range(s):
                tmp = grad([r[iatom]], g.ndata["pos"], retain_graph=iatom < s)[0]
                if tmp is not None:
                    hessian[iatom] = tmp.view(-1)

        if self.calc_stresses:
            volume = (
                torch.abs(torch.det(lattice.float())).half()
                if diep.float_th == torch.float16
                else torch.abs(torch.det(lattice))
            )
            sts = -grads[1]
            scale = 1.0 / volume * -160.21766208
            sts = [i * j for i, j in zip(sts, scale)] if sts.dim() == 3 else [sts * scale]
            stresses = torch.cat(sts)
        if self.debug_mode:
            return total_energies, grads[0], grads[1]

        if self.calc_site_wise:
            return total_energies, forces, stresses, hessian, site_wise

        return total_energies, forces, stresses, hessian
