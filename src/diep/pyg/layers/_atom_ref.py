"""Elemental reference offsets for the PyG backend."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch_geometric.utils import scatter

import diep


class AtomRef(nn.Module):
    """Get total property offset for a system."""

    def __init__(self, property_offset: torch.Tensor | None = None, max_z: int = 89) -> None:
        """
        Args:
            property_offset: per-element property offset. If given, its size overrides max_z.
            max_z: maximum atomic number.
        """
        super().__init__()
        if property_offset is None:
            property_offset = torch.zeros(max_z, dtype=diep.float_th)
        elif isinstance(property_offset, np.ndarray | list):
            property_offset = torch.tensor(property_offset, dtype=diep.float_th)

        self.max_z = property_offset.shape[-1]
        self.register_buffer("property_offset", property_offset)
        self.register_buffer("onehot", torch.eye(self.max_z))

    def get_feature_matrix(self, graphs: list) -> np.ndarray:
        """Count atoms of each element in every structure.

        Args:
            graphs: list of PyG graphs.

        Returns:
            (num_structures, num_elements) matrix of element counts.
        """
        features = torch.zeros(len(graphs), self.max_z, dtype=diep.float_th)
        for i, graph in enumerate(graphs):
            features[i] = torch.bincount(graph.node_type.long(), minlength=self.max_z)
        return features.cpu().numpy()

    def fit(self, graphs: list, properties: torch.Tensor) -> None:
        """Least-squares fit of the elemental reference values.

        Args:
            graphs: list of PyG graphs.
            properties: extensive property of each structure.
        """
        features = self.get_feature_matrix(graphs)
        self.property_offset = torch.tensor(
            np.linalg.pinv(features.T @ features) @ features.T @ np.array(properties), dtype=diep.float_th
        )

    def forward(self, data, state_attr: torch.Tensor | None = None):
        """Return the total property offset of each structure.

        Args:
            data: PyG graph (possibly a batch).
            state_attr: state attributes, used to select a row when property_offset is 2D.

        Returns:
            One offset per structure.
        """
        node_type = data.node_type.long()
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(node_type.numel(), dtype=torch.long, device=node_type.device)
        size = int(batch.max()) + 1 if batch.numel() else 1
        one_hot = self.onehot[node_type]

        if self.property_offset.ndim > 1:
            offsets = []
            for i in range(self.property_offset.size(dim=0)):
                per_atom = (self.property_offset[i].to(one_hot.device) * one_hot).sum(1)
                offsets.append(scatter(per_atom, batch, dim=0, dim_size=size, reduce="sum"))
            return torch.stack(offsets)[state_attr]

        per_atom = (self.property_offset.to(one_hot.device) * one_hot).sum(1)
        return scatter(per_atom, batch, dim=0, dim_size=size, reduce="sum")
