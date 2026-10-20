"""Three-Body interaction implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn

import diep
from diep.utils.maths import get_segment_indices_from_n, scatter_sum

if TYPE_CHECKING:
    import dgl


class ThreeBodyInteractions(nn.Module):
    """Include 3D interactions to the bond update."""

    def __init__(self, update_network_atom: nn.Module, update_network_bond: nn.Module, **kwargs):
        """
        Initialize ThreeBodyInteractions.

        Args:
            update_network_atom: MLP for node features in Eq.2
            update_network_bond: Gated-MLP for edge features in Eq.3
            **kwargs: Kwargs pass-through to nn.Module.__init__().
        """
        super().__init__(**kwargs)
        self.update_network_atom = update_network_atom
        self.update_network_bond = update_network_bond

    def forward(
        self,
        graph: dgl.DGLGraph,
        line_graph: dgl.DGLGraph,
        three_basis: torch.Tensor,
        three_cutoff: torch.Tensor,
        node_feat: torch.Tensor,
        edge_feat: torch.Tensor,
    ):
        """
        Forward function for ThreeBodyInteractions.

        Args:
            graph: dgl graph
            line_graph: line graph
            three_basis: three body basis expansion
            three_cutoff: cutoff radius
            node_feat: node features
            edge_feat: edge features

        Index-space precondition:
            ``line_graph`` node ids must be *parent-bond* ids of ``graph`` -- i.e.
            ``line_graph.num_nodes() == graph.num_edges()`` and node ``i`` is bond ``i``.
            ``create_line_graph`` guarantees this (``_remap_line_graph_to_bond_space``);
            ``diep.graph.compute.assert_lg_invariants`` checks it. The arithmetic below is
            correct as written and must not be "fixed": if it misbehaves, the caller handed
            in a line graph in pruned-bond index space.
        """
        # graph.edges()[1] is parent-bond indexed (length num_bonds); line_graph.edges()[1]
        # holds parent-bond ids, so this is the true end atom of the second bond of each triple.
        end_atom_indices = graph.edges()[1][line_graph.edges()[1]].to(diep.int_th)

        # Update node features using the atom update network
        updated_atoms = self.update_network_atom(node_feat)

        # Gather updated atom features for the end atoms
        end_atom_features = updated_atoms[end_atom_indices]

        # Compute the basis term
        basis = three_basis * end_atom_features

        # Reshape and compute weights based on the three-cutoff tensor.
        # three_cutoff is polynomial_cutoff(g.edata["bond_dist"], threebody_cutoff), built over
        # parent bonds; edge_indices holds parent-bond ids, so each triple picks up the envelope
        # of its own two bonds. Indexed with pruned-bond ids instead, a triple would often select
        # a bond beyond threebody_cutoff where the envelope is an exact zero, silently deleting it.
        three_cutoff = three_cutoff.unsqueeze(1)
        edge_indices = torch.stack(list(line_graph.edges()), dim=1)
        weights = three_cutoff[edge_indices].view(-1, 2)
        weights = weights.prod(dim=-1)

        # Compute the weighted basis
        basis = basis * weights[:, None]

        # Aggregate the new bonds using scatter_sum.
        # n_triple_ij has one entry per parent bond (zero for bonds in no triple), so segment_ids
        # are parent-bond ids and match the scatter width graph.num_edges(). The zeros make
        # get_segment_indices_from_n's empty-segment handling load-bearing.
        segment_ids = get_segment_indices_from_n(line_graph.ndata["n_triple_ij"])
        new_bonds = scatter_sum(
            basis.to(diep.float_th),
            segment_ids=segment_ids,
            num_segments=graph.num_edges(),
            dim=0,
        )

        # If no new bonds are generated, return the original edge features
        if new_bonds.shape[0] == 0:
            return edge_feat

        # Update edge features using the bond update network
        updated_edge_feat = edge_feat + self.update_network_bond(new_bonds)

        return updated_edge_feat
