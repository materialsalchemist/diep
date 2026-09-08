"""The DIEP model on PyTorch Geometric.

A translation of :mod:`diep.models._diep`. Submodule and parameter names are identical, so a
DGL checkpoint loads into this model unchanged and the two backends produce the same numbers.

The three-body line graph is built by :func:`diep.pyg.graph.compute.create_line_graph`,
which works in parent-bond index space throughout -- ``triple_index`` holds ids into
``data.edge_index`` and ``n_triple_ij`` has one entry per bond. Every tensor the forward pass
indexes with those ids (``edge_index``, ``bond_vec``, the polynomial three-body cutoff, the
three-body scatter width) is built over the same bonds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from pymatgen.core import Element
from torch import nn
from torch_geometric.utils import scatter

import diep
from diep.config import DEFAULT_ELEMENTS
from diep.layers._activations import ActivationFunction
from diep.layers._core import MLP, GatedMLP
from diep.models._core import MatGLModel
from diep.pyg.graph.compute import (
    compute_pair_vector_and_distance,
    create_line_graph,
    ensure_line_graph_compatibility,
)
from diep.pyg.layers import (
    DIEPIntegrator,
    EmbeddingBlock,
    M3GNetBlock,
    ReduceReadOut,
    Set2SetReadOut,
    ThreeBodyInteractions,
    WeightedAtomReadOut,
    WeightedReadOut,
)
from diep.utils.cutoff import polynomial_cutoff

if TYPE_CHECKING:
    from diep.pyg.graph.converters import GraphConverter


class DIEP(MatGLModel):
    """The main DIEP model, PyG backend."""

    __version__ = 1

    def __init__(
        self,
        element_types: tuple[str, ...] = DEFAULT_ELEMENTS,
        dim_node_embedding: int = 64,
        dim_edge_embedding: int = 64,
        dim_state_embedding: int = 0,
        ntypes_state: int | None = None,
        dim_state_feats: int | None = None,
        nblocks: int = 3,
        is_intensive: bool = True,
        readout_type: Literal["set2set", "weighted_atom", "reduce_atom"] = "weighted_atom",
        task_type: Literal["classification", "regression"] = "regression",
        cutoff: float = 5.0,
        threebody_cutoff: float = 4.0,
        units: int = 64,
        ntargets: int = 1,
        niters_set2set: int = 3,
        nlayers_set2set: int = 3,
        field: Literal["node_feat", "edge_feat"] = "node_feat",
        include_state: bool = False,
        activation_type: Literal["swish", "tanh", "sigmoid", "softplus2", "softexp"] = "swish",
        dropout: float | None = None,
        grid_half_length: float = 5.0,
        base_spacing: float = 1.0,
        gaussian_sigma: float = 1.0,
        integral_mode: Literal["sum", "grid"] = "grid",
        softening_epsilon: float = 0.5,
        use_effective_charge: bool = True,
        use_edges: bool | None = None,
        use_triplets: bool = True,
        **kwargs,
    ):
        """
        Args:
            element_types: elements appearing in the dataset.
            dim_node_embedding: number of embedded atomic features.
            dim_edge_embedding: number of edge features.
            dim_state_embedding: number of hidden neurons in the state embedding.
            ntypes_state: number of state labels.
            dim_state_feats: number of state features after the linear layer.
            nblocks: number of convolution blocks.
            is_intensive: whether the prediction is intensive.
            readout_type: `set2set`, `weighted_atom` (default) or `reduce_atom`.
            task_type: `classification` or `regression` (default).
            cutoff: cutoff radius of the graph.
            threebody_cutoff: cutoff radius for the three-body interaction.
            units: number of neurons in each MLP layer.
            ntargets: number of target properties.
            niters_set2set: number of set2set iterations.
            nlayers_set2set: number of set2set layers.
            field: "node_feat" or "edge_feat" for Set2Set and reduced readout.
            include_state: whether to include state features.
            activation_type: 'swish', 'tanh', 'sigmoid', 'softplus2' or 'softexp'.
            dropout: dropout probability applied in graph layers during training.
            grid_half_length: half-length of the 2D integration grid for DIEP.
            base_spacing: base grid spacing for DIEP integration.
            gaussian_sigma: width parameter for the Gaussian electron density.
            integral_mode: "sum" or "grid".
            softening_epsilon: softening parameter preventing 1/r singularities.
            use_effective_charge: use sqrt(Z) instead of Z.
            use_edges: if set, overrides triplet/line-graph usage.
            use_triplets: if False, skip triplet features and three-body interactions.
            **kwargs: for future flexibility. Not used at the moment.
        """
        super().__init__()
        self.save_args(locals(), kwargs)

        try:
            activation: nn.Module = ActivationFunction[activation_type].value()
        except KeyError:
            raise ValueError(
                f"Invalid activation type, please try using one of {[af.name for af in ActivationFunction]}"
            ) from None

        self.element_types = element_types or DEFAULT_ELEMENTS
        self.register_buffer(
            "atomic_number_table",
            torch.tensor([Element(el).Z for el in self.element_types], dtype=diep.float_th),
            persistent=False,
        )

        self.diep_integrator = DIEPIntegrator(
            grid_half_length=grid_half_length,
            base_spacing=base_spacing,
            sigma=gaussian_sigma,
            mode=integral_mode,
            softening_epsilon=softening_epsilon,
            use_effective_charge=use_effective_charge,
        )
        degree = self.diep_integrator.edge_dim
        degree_rbf = degree

        self.embedding = EmbeddingBlock(
            degree_rbf=degree_rbf,
            dim_node_embedding=dim_node_embedding,
            dim_edge_embedding=dim_edge_embedding,
            ntypes_node=len(element_types),
            ntypes_state=ntypes_state,
            dim_state_feats=dim_state_feats,
            include_state=include_state,
            dim_state_embedding=dim_state_embedding,
            activation=activation,
        )

        self.three_body_interactions = nn.ModuleList(
            [
                ThreeBodyInteractions(
                    update_network_atom=MLP(
                        dims=[dim_node_embedding, degree], activation=nn.Sigmoid(), activate_last=True
                    ),
                    update_network_bond=GatedMLP(in_feats=degree, dims=[dim_edge_embedding], use_bias=False),
                )
                for _ in range(nblocks)
            ]
        )

        dim_state_feats = dim_state_embedding

        self.graph_layers = nn.ModuleList(
            [
                M3GNetBlock(
                    degree=degree_rbf,
                    activation=activation,
                    conv_hiddens=[units, units],
                    dim_node_feats=dim_node_embedding,
                    dim_edge_feats=dim_edge_embedding,
                    dim_state_feats=dim_state_feats,
                    include_state=include_state,
                    dropout=dropout,
                )
                for _ in range(nblocks)
            ]
        )

        if is_intensive:
            input_feats = dim_node_embedding if field == "node_feat" else dim_edge_embedding
            if readout_type == "set2set":
                self.readout = Set2SetReadOut(
                    in_feats=input_feats, n_iters=niters_set2set, n_layers=nlayers_set2set, field=field
                )
                readout_feats = 2 * input_feats + dim_state_feats if include_state else 2 * input_feats
            elif readout_type == "weighted_atom":
                self.readout = WeightedAtomReadOut(in_feats=input_feats, dims=[units, units], activation=activation)
                readout_feats = units + dim_state_feats if include_state else units
            else:
                self.readout = ReduceReadOut("mean", field=field)
                readout_feats = input_feats + dim_state_feats if include_state else input_feats

            self.final_layer = MLP([readout_feats, units, units, ntargets], activation, activate_last=False)
            if task_type == "classification":
                self.sigmoid = nn.Sigmoid()
        else:
            if task_type == "classification":
                raise ValueError("Classification task cannot be extensive.")
            self.final_layer = WeightedReadOut(
                in_feats=dim_node_embedding, dims=[units, units], num_targets=ntargets
            )

        self.n_blocks = nblocks
        self.units = units
        self.cutoff = cutoff
        self.threebody_cutoff = threebody_cutoff
        self.include_state = include_state
        self.task_type = task_type
        self.is_intensive = is_intensive
        if use_edges is not None:
            use_triplets = use_edges
        self.use_triplets = use_triplets
        self.use_edges = use_triplets

    def forward(self, g, state_attr: torch.Tensor | None = None, l_g=None, return_all_layer_output: bool = False):
        """Message passing over the graph, returning the target property.

        Args:
            g: PyG graph (or batch) for the structures.
            state_attr: state attributes.
            l_g: ignored; the line graph lives on ``g`` as ``triple_index`` / ``n_triple_ij``.
                Accepted so the call signature matches the DGL model.
            return_all_layer_output: return the output of every DIEP layer rather than just
                the final one.

        Returns:
            The predicted property, or a dict of per-layer outputs.
        """
        del l_g  # the PyG line graph is carried on `g` itself
        node_types = g.node_type.long()
        bond_vec, bond_dist = compute_pair_vector_and_distance(g)
        g.bond_vec = bond_vec
        g.bond_dist = bond_dist

        use_edges = self.use_edges
        if use_edges:
            # builds the line graph if absent, validates it against this graph's bond count
            # if it came from a cache; either way `g.triple_index` ends up in parent-bond space
            ensure_line_graph_compatibility(g, self.threebody_cutoff)

        atomic_table = self.atomic_number_table
        if atomic_table.device != node_types.device:
            atomic_table = atomic_table.to(node_types.device)
        atomic_numbers = atomic_table[node_types].to(diep.float_th)

        bond_features, triplet_features = self.diep_integrator(g, atomic_numbers, compute_triplets=use_edges)
        g.rbf = bond_features
        if use_edges:
            three_body_basis = triplet_features
            three_body_cutoff = polynomial_cutoff(g.bond_dist, self.threebody_cutoff)

        node_feat, edge_feat, state_feat = self.embedding(node_types, g.rbf, state_attr)
        fea_dict = {"diep_embedding": g.rbf}
        for i in range(self.n_blocks):
            if use_edges:
                edge_feat = self.three_body_interactions[i](
                    g, three_body_basis, three_body_cutoff, node_feat, edge_feat
                )
            edge_feat, node_feat, state_feat = self.graph_layers[i](g, edge_feat, node_feat, state_feat)
            fea_dict[f"gc_{i + 1}"] = {
                "node_feat": node_feat,
                "edge_feat": edge_feat,
                "state_feat": state_feat,
            }
        g.node_feat = node_feat
        g.edge_feat = edge_feat

        if self.is_intensive:
            field_vec = self.readout(g)
            readout_vec = torch.hstack([field_vec, state_feat]) if self.include_state else field_vec
            fea_dict["readout"] = readout_vec
            output = self.final_layer(readout_vec)
            if self.task_type == "classification":
                output = self.sigmoid(output)
        else:
            atomic_properties = self.final_layer(g)
            g.atomic_properties = atomic_properties
            fea_dict["readout"] = atomic_properties
            batch = getattr(g, "batch", None)
            if batch is None:
                batch = torch.zeros(atomic_properties.size(0), dtype=torch.long, device=atomic_properties.device)
            size = int(batch.max()) + 1 if batch.numel() else 1
            output = scatter(atomic_properties, batch, dim=0, dim_size=size, reduce="sum")

        fea_dict["final"] = output
        if return_all_layer_output:
            return fea_dict
        return torch.squeeze(output)

    def predict_structure(
        self,
        structure,
        state_feats: torch.Tensor | None = None,
        graph_converter: GraphConverter | None = None,
        output_layers: list | None = None,
        return_features: bool = False,
    ):
        """Featurize or predict the property of a structure.

        Args:
            structure: input crystal or molecule.
            state_feats: graph attributes.
            graph_converter: object implementing ``get_graph``. Defaults to Structure2Graph
                at this model's cutoff.
            output_layers: names of layers to return when ``return_features`` is True.
            return_features: whether to return layer outputs instead of the final value.

        Returns:
            The predicted property, or a dict of layer outputs.
        """
        from diep.pyg.graph.converters import Structure2Graph  # noqa: PLC0415  (circular import)

        allowed = ["diep_embedding", *[f"gc_{i + 1}" for i in range(self.n_blocks)], "readout", "final"]
        if output_layers is None:
            output_layers = allowed
        elif not isinstance(output_layers, list) or set(output_layers).difference(allowed):
            raise ValueError(f"Invalid output_layers, it must be a sublist of {allowed}.")

        if graph_converter is None:
            graph_converter = Structure2Graph(element_types=self.element_types, cutoff=self.cutoff)
        g, lat, state_feats_default = graph_converter.get_graph(structure)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        if state_feats is None:
            state_feats = torch.tensor(state_feats_default)
        if self.use_edges:
            bond_vec, bond_dist = compute_pair_vector_and_distance(g)
            g.bond_vec, g.bond_dist = bond_vec, bond_dist
            create_line_graph(g, self.threebody_cutoff)
        if return_features:
            return {k: v for k, v in self(g=g, state_attr=state_feats, return_all_layer_output=True).items()
                    if k in output_layers}
        return self(g=g, state_attr=state_feats)
