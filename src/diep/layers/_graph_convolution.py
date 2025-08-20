"""Graph convolution layer (GCL) implementations."""

from __future__ import annotations

import dgl
import dgl.function as fn
import torch
from torch import Tensor, nn
from torch.nn import Dropout, Identity, Module

from diep.layers._core import MLP, GatedMLP


class DIEPGraphConv(Module):
    """A M3GNet graph convolution layer in DGL."""

    def __init__(
        self,
        include_states: bool,
        edge_update_func: Module,
        edge_weight_func: Module,
        node_update_func: Module,
        node_weight_func: Module,
        state_update_func: Module | None,
    ):
        """Parameters:
        include_states (bool): Whether including state
        edge_update_func (Module): Update function for edges (Eq. 4)
        edge_weight_func (Module): Weight function for radial basis functions (Eq. 4)
        node_update_func (Module): Update function for nodes (Eq. 5)
        node_weight_func (Module): Weight function for radial basis functions (Eq. 5)
        state_update_func (Module): Update function for state feats (Eq. 6).
        """
        super().__init__()
        self.include_states = include_states
        self.edge_update_func = edge_update_func
        self.edge_weight_func = edge_weight_func
        self.node_update_func = node_update_func
        self.node_weight_func = node_weight_func
        self.state_update_func = state_update_func

    @staticmethod
    def from_dims(
        degree,
        include_states,
        edge_dims: list[int],
        node_dims: list[int],
        state_dims: list[int] | None,
        activation: Module,
    ) -> DIEPGraphConv:
        """M3GNetGraphConv initialization.

        Args:
            degree (int): max_n*max_l
            include_states (bool): whether including state or not
            edge_dims (list): NN architecture for edge update function
            node_dims (list): NN architecture for node update function
            state_dims (list): NN architecture for state update function
            activation (nn.Nodule): activation function

        Returns:
        M3GNetGraphConv (class)
        """
        edge_update_func = GatedMLP(in_feats=edge_dims[0], dims=edge_dims[1:])
        edge_weight_func = nn.Linear(
            in_features=degree, out_features=edge_dims[-1], bias=False
        )

        node_update_func = GatedMLP(in_feats=node_dims[0], dims=node_dims[1:])
        node_weight_func = nn.Linear(
            in_features=degree, out_features=node_dims[-1], bias=False
        )
        attr_update_func = MLP(state_dims, activation, activate_last=True) if include_states else None  # type: ignore
        return DIEPGraphConv(
            include_states,
            edge_update_func,
            edge_weight_func,
            node_update_func,
            node_weight_func,
            attr_update_func,
        )

    def _edge_udf(self, edges: dgl.udf.EdgeBatch):
        """Edge update functions.

        Args:
        edges (DGL graph): edges in dgl graph

        Returns:
        mij: message passing between node i and j
        """
        vi = edges.src["v"]
        vj = edges.dst["v"]
        u = None
        if self.include_states:
            u = edges.src["u"]
        eij = edges.data.pop("e")
        rbf = edges.data["rbf"]
        inputs = (
            torch.hstack([vi, vj, eij, u])
            if self.include_states
            else torch.hstack([vi, vj, eij])
        )
        mij = {"mij": self.edge_update_func(inputs) * self.edge_weight_func(rbf)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) -> Tensor:
        """Perform edge update.

        Args:
        graph: DGL graph

        Returns:
        edge_update: edge features update
        """
        graph.apply_edges(self._edge_udf)
        edge_update = graph.edata.pop("mij")
        return edge_update

    def node_update_(self, graph: dgl.DGLGraph, state_feat: Tensor) -> Tensor:
        """Perform node update.

        Args:
            graph: DGL graph
            state_feat: State attributes

        Returns:
            node_update: node features update
        """
        eij = graph.edata["e"]
        src_id = graph.edges()[0]
        vi = graph.ndata["v"][src_id]
        dst_id = graph.edges()[1]
        vj = graph.ndata["v"][dst_id]
        rbf = graph.edata["rbf"]
        if self.include_states:
            u = dgl.broadcast_edges(graph, state_feat)
            inputs = torch.hstack([vi, vj, eij, u])
        else:
            inputs = torch.hstack([vi, vj, eij])
        graph.edata["mess"] = self.node_update_func(inputs) * self.node_weight_func(rbf)
        graph.update_all(fn.copy_e("mess", "mess"), fn.sum("mess", "ve"))
        node_update = graph.ndata.pop("ve")
        return node_update

    def state_update_(self, graph: dgl.DGLGraph, state_feat: Tensor) -> Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: DGL graph
            state_feat: graph features

        Returns:
        state_update: state_features update
        """
        u = state_feat
        uv = dgl.readout_nodes(graph, feat="v", op="mean")
        inputs = torch.hstack([u, uv])
        state_feat = self.state_update_func(inputs)  # type: ignore
        return state_feat

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Perform sequence of edge->node->states updates.

        Args:
            graph: Input graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: Graph attributes (global state).

        Returns:
            (edge features, node features, graph attributes)
        """
        with graph.local_scope():
            graph.edata["e"] = edge_feat
            graph.ndata["v"] = node_feat
            if self.include_states:
                graph.ndata["u"] = dgl.broadcast_nodes(graph, state_feat)

            edge_update = self.edge_update_(graph)
            graph.edata["e"] = edge_feat + edge_update
            node_update = self.node_update_(graph, state_feat)
            graph.ndata["v"] = node_feat + node_update
            if self.include_states:
                state_feat = self.state_update_(graph, state_feat)

        return edge_feat + edge_update, node_feat + node_update, state_feat


class DIEPBlock(Module):
    """A M3GNet block comprising a sequence of update operations."""

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
            degree: Number of radial basis functions
            activation: activation
            dim_node_feats: Number of node features
            dim_edge_feats: Number of edge features
            dim_state_feats: Number of state features
            conv_hiddens: Dimension of hidden layers
            activation: Activation type
            include_state: Including state features or not
            dropout: Probability of an element to be zero in dropout layer.
        """
        super().__init__()

        self.activation = activation

        # compute input sizes
        if include_state:
            edge_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats  # type: ignore
            node_in = 2 * dim_node_feats + dim_edge_feats + dim_state_feats  # type: ignore
            attr_in = dim_node_feats + dim_state_feats  # type: ignore
            self.conv = DIEPGraphConv.from_dims(
                degree,
                include_state,
                edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
                node_dims=[node_in, *conv_hiddens, dim_node_feats],
                state_dims=[attr_in, *conv_hiddens, dim_state_feats],  # type: ignore
                activation=self.activation,
            )
        else:
            edge_in = 2 * dim_node_feats + dim_edge_feats  # 2*NDIM+EDIM
            node_in = 2 * dim_node_feats + dim_edge_feats  # 2*NDIM+EDIM
            self.conv = DIEPGraphConv.from_dims(
                degree,
                include_state,
                edge_dims=[edge_in, *conv_hiddens, dim_edge_feats],
                node_dims=[node_in, *conv_hiddens, dim_node_feats],
                state_dims=None,  # type: ignore
                activation=self.activation,
            )

        self.dropout = Dropout(dropout) if dropout else None

    def forward(
        self,
        graph: dgl.DGLGraph,
        edge_feat: Tensor,
        node_feat: Tensor,
        state_feat: Tensor,
    ) -> tuple:
        """
        Args:
            graph: DGL graph
            edge_feat: Edge features
            node_feat: Node features
            state_feat: State features.

        Returns:
            A tuple of updated features
        """
        edge_feat, node_feat, state_feat = self.conv(
            graph, edge_feat, node_feat, state_feat
        )

        if self.dropout:
            edge_feat = self.dropout(edge_feat)  # pylint: disable=E1102
            node_feat = self.dropout(node_feat)  # pylint: disable=E1102
            if state_feat is not None:
                state_feat = self.dropout(state_feat)  # pylint: disable=E1102
        return edge_feat, node_feat, state_feat
