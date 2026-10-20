"""Three-body interactions on PyG graphs."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.utils import scatter

import diep


class ThreeBodyInteractions(nn.Module):
    """Include 3-body interactions in the bond update.

    Parameter names match :class:`diep.layers._three_body.ThreeBodyInteractions` exactly, so
    a DGL checkpoint loads into this module unchanged.
    """

    def __init__(self, update_network_atom: nn.Module, update_network_bond: nn.Module, **kwargs):
        """
        Args:
            update_network_atom: MLP for node features in Eq.2.
            update_network_bond: Gated-MLP for edge features in Eq.3.
            **kwargs: Kwargs pass-through to nn.Module.__init__().
        """
        super().__init__()
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(
        self,
        data,
        three_basis: torch.Tensor,
        three_cutoff: torch.Tensor,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        """Update bond features with the three-body channel.

        Args:
            data: graph carrying ``edge_index``, ``triple_index`` and ``n_triple_ij``.
            three_basis: three-body basis expansion, one row per triple.
            three_cutoff: polynomial cutoff envelope, one entry per *bond*.
            node_feat: node features.
            edge_feat: edge features.

        Index-space precondition:
            ``data.triple_index`` holds parent-bond ids, i.e. ids into ``data.edge_index``,
            and ``data.n_triple_ij`` has one entry per bond. ``create_line_graph``
            guarantees both; ``assert_lg_invariants`` checks them.
        """
        num_bonds = int(data.edge_index.size(1))
        first, second = data.triple_index

        # edge_index[1] is parent-bond indexed, and `second` holds parent-bond ids, so this
        # is the true end atom of the second bond of each triple.
        end_atom_indices = data.edge_index[1][second]
        updated_atoms = self.update_network_atom(node_feat)
        basis = three_basis * updated_atoms[end_atom_indices]

        # three_cutoff is built over parent bonds, and first/second are parent-bond ids, so
        # each triple picks up the envelope of its own two bonds. Indexed with pruned-bond
        # ids instead, a triple would often select a bond beyond threebody_cutoff where the
        # envelope is an exact zero, silently deleting the triple rather than attenuating it.
        weights = three_cutoff[first] * three_cutoff[second]
        basis = basis * weights[:, None]

        # `first` is the parent-bond id each triple contributes to, and it is sorted
        # ascending, so this is the same aggregation the DGL path performs via
        # get_segment_indices_from_n(n_triple_ij).
        new_bonds = scatter(
            basis.to(diep.float_th), first.long(), dim=0, dim_size=num_bonds, reduce="sum"
        )
        if new_bonds.shape[0] == 0:
            return edge_feat
        return edge_feat + self.update_network_bond(new_bonds)
