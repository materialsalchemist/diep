"""Embedding node, edge and optional state attributes."""

from __future__ import annotations

import dgl
import dgl.function as fn
import torch
from torch import nn

import diep
from diep.layers._core import MLP
from diep.utils.cutoff import cosine_cutoff
# from diep import device
# torch.set_default_device(device)


class EmbeddingBlock(nn.Module):
    """Embedding block for generating node, bond and state features."""

    def __init__(
        self,
        degree_rbf: int,
        activation: nn.Module,
        dim_node_embedding: int,
        dim_edge_embedding: int | None = None,
        dim_state_feats: int | None = None,
        ntypes_node: int | None = None,
        include_state: bool = False,
        ntypes_state: int | None = None,
        dim_state_embedding: int | None = None,
    ):
        """
        Args:
            degree_rbf (int): number of rbf
            activation (nn.Module): activation type
            dim_node_embedding (int): dimensionality of node features
            dim_edge_embedding (int): dimensionality of edge features
            dim_state_feats: dimensionality of state features
            ntypes_node: number of node labels
            include_state: Whether to include state embedding
            ntypes_state: number of state labels
            dim_state_embedding: dimensionality of state embedding.
        """
        super().__init__()
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.dim_node_embedding = dim_node_embedding
        self.dim_edge_embedding = dim_edge_embedding
        self.dim_state_feats = dim_state_feats
        self.ntypes_node = ntypes_node
        self.dim_state_embedding = dim_state_embedding
        self.activation = activation
        if ntypes_state and dim_state_embedding is not None:
            self.layer_state_embedding = nn.Embedding(ntypes_state, dim_state_embedding)  # type: ignore
        elif dim_state_feats is not None:
            self.layer_state_embedding = nn.Sequential(
                nn.LazyLinear(dim_state_feats, bias=False, dtype=diep.float_th),
                activation,
            )
        if ntypes_node is not None:
            self.layer_node_embedding = nn.Embedding(ntypes_node, dim_node_embedding)
        else:
            self.layer_node_embedding = nn.Sequential(
                nn.LazyLinear(dim_node_embedding, bias=False, dtype=diep.float_th),
                activation,
            )
        if dim_edge_embedding is not None:
            dim_edges = [degree_rbf, dim_edge_embedding]
            self.layer_edge_embedding = MLP(dim_edges, activation=activation, activate_last=True)

    def forward(self, node_attr, edge_attr, state_attr):
        """Output embedded features.

        Args:
            node_attr: node attribute
            edge_attr: edge attribute
            state_attr: state attribute

        Returns:
            node_feat: embedded node features
            edge_feat: embedded edge features
            state_feat: embedded state features
        """
        if self.ntypes_node is not None:
            node_feat = self.layer_node_embedding(node_attr)
        else:
            node_feat = self.layer_node_embedding(node_attr.to(diep.float_th))
        if self.dim_edge_embedding is not None:
            edge_feat = self.layer_edge_embedding(edge_attr.to(diep.float_th))
        else:
            edge_feat = edge_attr
        if self.include_state is True:
            if self.ntypes_state and self.dim_state_embedding is not None:
                state_feat = self.layer_state_embedding(state_attr)
            elif self.dim_state_feats is not None:
                state_attr = torch.unsqueeze(state_attr, 0)
                state_feat = self.layer_state_embedding(state_attr.to(diep.float_th))
            else:
                state_feat = state_attr
        else:
            state_feat = None
        return node_feat, edge_feat, state_feat


class NeighborEmbedding(nn.Module):
    def __init__(
        self,
        ntypes_node: int,
        hidden_channels: int,
        num_rbf: int,
        cutoff: float,
        dtype: torch.dtype = diep.float_th,
    ):
        """
        The ET architecture assigns two  learned vectors to each atom type
        zi. One  is used to  encode information  specific to an  atom, the
        other (this  class) takes  the role  of a  neighborhood embedding.
        The neighborhood embedding, which is  an embedding of the types of
        neighboring atoms, is multiplied by a distance filter.


        This embedding allows  the network to store  information about the
        interaction of atom pairs.

        See eq. 3 in https://arxiv.org/pdf/2202.02541.pdf for more details.
        """
        super().__init__()
        self.embedding = nn.Embedding(ntypes_node, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels, dtype=dtype)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels, dtype=dtype)
        self.cutoff = cutoff

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(
        self,
        z: torch.Tensor,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z (Tensor): Atomic numbers of shape [num_nodes].
            node_feat (Tensor): graph-convoluted node features [num_nodes, hidden_channels].
            edge_index (Tensor): Graph connectivity (list of neighbor pairs) with shape [2, num_edges].
            edge_weight (Tensor): Edge weight vector of shape [num_edges].
            edge_attr (Tensor): Edge attribute matrix of shape [num_edges, num_rbf].

        Returns:
            x_neighbors (Tensor): The embedding of the neighbors of each atom of shape [num_nodes, hidden_channels].
        """
        C = cosine_cutoff(edge_weight, self.cutoff)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        msg = W * x_neighbors.index_select(0, edge_index[1])
        x_neighbors = torch.zeros(
            node_feat.shape[0],
            node_feat.shape[1],
            dtype=node_feat.dtype,
            device=node_feat.device,
        ).index_add(0, edge_index[0], msg)
        x_neighbors = self.combine(torch.cat([node_feat, x_neighbors], dim=1))
        return x_neighbors
