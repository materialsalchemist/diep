"""M3GNet graph convolution on PyG graphs.

A direct translation of :mod:`diep.layers._graph_convolution`: the same networks in the same
order, with DGL's ``apply_edges`` / ``update_all`` replaced by gather-and-scatter. Parameter
names are identical, so DGL checkpoints load unchanged.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import Dropout, Module
from torch_geometric.utils import scatter

from diep.layers._core import MLP, GatedMLP


class M3GNetGraphConv(Module):
    """M3GNet graph convolution layer implemented with PyG-style scatter primitives."""

    def __init__(
        self,
        include_state: bool,
        edge_update_func: Module,
        edge_weight_func: Module,
        node_update_func: Module,
        node_weight_func: Module,
        state_update_func: Module | None,
    ) -> None:
        """
        Args:
            include_state: Whether global state features are present.
            edge_update_func: Edge update network.
            edge_weight_func: Linear projection producing edge weights from radial basis features.
            node_update_func: Node update network.
            node_weight_func: Linear projection producing node weights from radial basis features.
            state_update_func: State update network (optional when `include_state` is False).
        """
        super().__init__()
        self.include_state = include_state
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @staticmethod
    def from_dims(
        degree: int,
        include_state: bool,
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int] | None,
        activation: Module,
    ) -> M3GNetGraphConv:
        """Construct a graph convolution layer from network dimensions."""
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        edge_weight_func = nn.Linear(in_features=degree, out_features=edge_dims[-1], bias=False)
        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = nn.Linear(in_features=degree, out_features=node_dims[-1], bias=False)
        state_update_func = MLP(state_dims, activation, activate_last=True) if include_state else None
        return M3GNetGraphConv(
            include_state,
            edge_update_func,
            edge_weight_func,
            node_update_func,
            node_weight_func,
            state_update_func,
        )

    def _inputs(self, data, edge_feat, node_feat, state_feat):
        src, dst = data.edge_index
        vi = node_feat[src]
        vj = node_feat[dst]
        if self.include_state and state_feat is not None:
            # one state row per structure, broadcast to that structure's edges
            u = state_feat[data.batch[src]] if hasattr(data, "batch") and data.batch is not None else (
                state_feat.expand(vi.size(0), -1)
            )
            return torch.hstack([vi, vj, edge_feat, u])
        return torch.hstack([vi, vj, edge_feat])

    def edge_update_(self, data, edge_feat: Tensor, node_feat: Tensor, state_feat: Tensor | None) -> Tensor:
        """Apply the edge update and return the edge-feature increment."""
        inputs = self._inputs(data, edge_feat, node_feat, state_feat)
        return self.edge_update_func(inputs) * self.edge_weight_func(data.rbf)

    def node_update_(self, data, edge_feat: Tensor, node_feat: Tensor, state_feat: Tensor | None) -> Tensor:
        """Apply the node update and return the aggregated node-feature increment."""
        _, dst = data.edge_index
        inputs = self._inputs(data, edge_feat, node_feat, state_feat)
        mess = self.node_update_func(inputs) * self.node_weight_func(data.rbf)
        # DGL's update_all(copy_e("mess"), sum("mess", "ve")) aggregates each edge's message
        # at the edge's *destination* node, so the scatter index is edge_index[1].
        return scatter(mess, dst.long(), dim=0, dim_size=node_feat.size(0), reduce="sum")

    def state_update_(self, data, node_feat: Tensor, state_feat: Tensor) -> Tensor:
        """Update the global state features."""
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(
            node_feat.size(0), dtype=torch.long, device=node_feat.device
        )
        uv = scatter(node_feat, batch, dim=0, dim_size=state_feat.size(0), reduce="mean")
        return self.state_update_func(torch.hstack([state_feat, uv]))

    def forward(
        self, data, edge_feat: Tensor, node_feat: Tensor, state_feat: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Perform edge, node, and optional state updates."""
        edge_update = self.edge_update_(data, edge_feat, node_feat, state_feat)
        new_edge_feat = edge_feat + edge_update
        node_update = self.node_update_(data, new_edge_feat, node_feat, state_feat)
        new_node_feat = node_feat + node_update
        if self.include_state and state_feat is not None:
            state_feat = self.state_update_(data, new_node_feat, state_feat)
        return new_edge_feat, new_node_feat, state_feat


class M3GNetBlock(Module):
    """Stacked graph convolution block following the M3GNet design."""

    def __init__(
        self,
        degree: int,
        activation: Module,
        conv_hiddens: list[int],
        dim_node_feats: int,
        dim_edge_feats: int,
        dim_state_feats: int = 0,
        include_state: bool = False,
        dropout: float | None = None,
    ) -> None:
        """
        Args:
            degree: width of the radial basis feature.
            activation: activation module.
            conv_hiddens: hidden dimensions of the update networks.
            dim_node_feats: node feature dimension.
            dim_edge_feats: edge feature dimension.
            dim_state_feats: state feature dimension.
            include_state: whether global state features are used.
            dropout: dropout probability, or None.
        """
        super().__init__()
        self.include_state = include_state
        self.activation = activation

        if include_state:
            edge_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats
            node_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats
            state_dims = [dim_node_feats + dim_state_feats, *conv_hiddens, dim_state_feats]
        else:
            edge_in = 2 * dim_node_feats + dim_edge_feats
            node_in = 2 * dim_node_feats + dim_edge_feats
            state_dims = None

        self.conv = M3GNetGraphConv.from_dims(
            degree=degree,
            include_state=include_state,
            edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
            node_dims=[node_in, *conv_hiddens, dim_node_feats],
            state_dims=state_dims,
            activation=self.activation,
        )
        self.dropout = Dropout(dropout) if dropout else None

    def forward(
        self, data, edge_feat: Tensor, node_feat: Tensor, state_feat: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        """Run a forward pass through the block."""
        edge_feat, node_feat, state_feat = self.conv(data, edge_feat, node_feat, state_feat)
        if self.dropout:
            edge_feat = self.dropout(edge_feat)
            node_feat = self.dropout(node_feat)
            if state_feat is not None:
                state_feat = self.dropout(state_feat)
        return edge_feat, node_feat, state_feat
