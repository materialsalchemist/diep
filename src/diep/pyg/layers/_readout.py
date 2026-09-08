"""Readout layers for the PyG backend.

Translations of :mod:`diep.layers._readout`. Parameter names match the DGL versions, so
checkpoints transfer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn
from torch_geometric.utils import scatter, softmax

from diep.layers._core import MLP, GatedMLP

if TYPE_CHECKING:
    from collections.abc import Sequence


def _batch_of(data, num_rows: int, device) -> torch.Tensor:
    """Return the structure index of each node, defaulting to a single structure."""
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros(num_rows, dtype=torch.long, device=device)
    return batch


def _edge_batch_of(data) -> torch.Tensor:
    """Return the structure index of each bond."""
    src = data.edge_index[0]
    batch = getattr(data, "batch", None)
    if batch is None:
        return torch.zeros(src.numel(), dtype=torch.long, device=src.device)
    return batch[src]


class EdgeSet2Set(nn.Module):
    """Set2Set pooling over edges, mirroring :class:`diep.layers._core.EdgeSet2Set`."""

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) -> None:
        """
        Args:
            input_dim: size of each input sample.
            n_iters: number of Set2Set iterations.
            n_layers: number of recurrent layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialise learnable parameters."""
        self.lstm.reset_parameters()

    def forward(self, data, feat: torch.Tensor) -> torch.Tensor:
        """Pool edge features into one vector per structure."""
        edge_batch = _edge_batch_of(data)
        batch_size = int(edge_batch.max()) + 1 if edge_batch.numel() else 1
        h = (
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
            feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
        )
        q_star = feat.new_zeros(batch_size, self.output_dim)
        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)
            e = (feat * q[edge_batch]).sum(dim=-1, keepdim=True)
            alpha = softmax(e, edge_batch, num_nodes=batch_size)
            readout = scatter(feat * alpha, edge_batch, dim=0, dim_size=batch_size, reduce="sum")
            q_star = torch.cat([q, readout], dim=-1)
        return q_star


class Set2SetReadOut(nn.Module):
    """The Set2Set readout function."""

    def __init__(self, in_feats: int, n_iters: int, n_layers: int, field: Literal["node_feat", "edge_feat"]):
        """
        Args:
            in_feats: length of the input feature vector.
            n_iters: number of LSTM steps.
            n_layers: number of layers.
            field: which field of the graph to read out.
        """
        super().__init__()
        self.field = field
        self.n_iters = n_iters
        self.n_layers = n_layers
        if field == "node_feat":
            from torch_geometric.nn.aggr import Set2Set  # noqa: PLC0415

            self.set2set = Set2Set(in_feats, n_iters, n_layers)
        elif field == "edge_feat":
            self.set2set = EdgeSet2Set(in_feats, n_iters, n_layers)
        else:
            raise ValueError("Field must be node_feat or edge_feat")

    def forward(self, data):
        """Read out one vector per structure."""
        if self.field == "node_feat":
            batch = _batch_of(data, data.node_feat.size(0), data.node_feat.device)
            return self.set2set(data.node_feat, batch)
        return self.set2set(data, data.edge_feat)


class ReduceReadOut(nn.Module):
    """Reduce atom or bond attributes into lower dimensional tensors as readout."""

    def __init__(self, op: str = "mean", field: Literal["node_feat", "edge_feat"] = "node_feat"):
        """
        Args:
            op: reduction, e.g. "mean" or "sum".
            field: which field of the graph to reduce.
        """
        super().__init__()
        self.op = op
        self.field = field

    def forward(self, data):
        """Reduce over the nodes or edges of each structure."""
        if self.field == "node_feat":
            feat = data.node_feat
            index = _batch_of(data, feat.size(0), feat.device)
        else:
            feat = data.edge_feat
            index = _edge_batch_of(data)
        size = int(index.max()) + 1 if index.numel() else 1
        return scatter(feat, index, dim=0, dim_size=size, reduce=self.op)


class WeightedReadOut(nn.Module):
    """Feed node features into a Gated MLP as readout for atomic properties."""

    def __init__(self, in_feats: int, dims: Sequence[int], num_targets: int):
        """
        Args:
            in_feats: input node feature width.
            dims: hidden dimensions of the gated MLP.
            num_targets: number of target properties.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims, num_targets]
        self.gated = GatedMLP(in_feats=in_feats, dims=self.dims, activate_last=False)

    def forward(self, data):
        """Return one property vector per atom."""
        return self.gated(data.node_feat)


class WeightedAtomReadOut(nn.Module):
    """Weighted atom readout for graph properties."""

    def __init__(self, in_feats: int, dims: Sequence[int], activation: nn.Module):
        """
        Args:
            in_feats: input node feature width.
            dims: hidden dimensions of the MLP.
            activation: activation for the MLP.
        """
        super().__init__()
        self.dims = [in_feats, *dims]
        self.activation = activation
        self.mlp = MLP(dims=self.dims, activation=self.activation, activate_last=True)
        self.weight = nn.Sequential(nn.Linear(in_feats, 1), nn.Sigmoid())

    def forward(self, data):
        """Return one weighted-sum vector per structure."""
        node_feat = data.node_feat
        h = self.mlp(node_feat)
        w = self.weight(node_feat)
        batch = _batch_of(data, node_feat.size(0), node_feat.device)
        size = int(batch.max()) + 1 if batch.numel() else 1
        # dgl.sum_nodes(g, "h", "w") weights each node's h by its scalar w before summing
        return scatter(h * w, batch, dim=0, dim_size=size, reduce="sum")
