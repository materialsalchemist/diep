"""Core MLP building blocks, vendored DGL-free.

Only :class:`MLP` and :class:`GatedMLP` are kept from the DGL-era ``diep.layers._core``:
they are plain ``torch.nn`` modules. The rest of that module (``MLP_norm``,
``EdgeSet2Set``, the gated-equivariant blocks) is either unused here or graph-bound --
the PyG backend has its own ``EdgeSet2Set`` in :mod:`diep_pyg.layers._readout`.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from torch import Tensor, nn
from torch.nn import Linear, ModuleList

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


class MLP(nn.Module):
    """An implementation of a multi-layer perceptron."""

    def __init__(
        self,
        dims: Sequence[int],
        activation: Callable[[Tensor], Tensor] | None = None,
        activate_last: bool = False,
        bias_last: bool = True,
    ) -> None:
        """:param dims: Dimensions of each layer of MLP.
        :param activation: Activation function.
        :param activate_last: Whether to apply activation to last layer.
        :param bias_last: Whether to apply bias to last layer.
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = ModuleList()

        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            if i < self._depth - 1:
                self.layers.append(Linear(in_dim, out_dim, bias=True))

                if activation is not None:
                    self.layers.append(activation)  # type: ignore
            else:
                self.layers.append(Linear(in_dim, out_dim, bias=bias_last))

                if activation is not None and activate_last:
                    self.layers.append(activation)  # type: ignore

    def __repr__(self):
        dims = []

        for layer in self.layers:
            if isinstance(layer, Linear):
                dims.append(f"{layer.in_features} \u2192 {layer.out_features}")
            else:
                dims.append(layer.__class__.__name__)

        return f"MLP({', '.join(dims)})"

    @property
    def last_linear(self) -> Linear | None:
        """:return: The last linear layer."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer
        raise RuntimeError

    @property
    def depth(self) -> int:
        """Returns depth of MLP."""
        return self._depth

    @property
    def in_features(self) -> int:
        """Return input features of MLP."""
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        """Returns output features of MLP."""
        for layer in reversed(self.layers):
            if isinstance(layer, Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs):
        """Applies all layers in turn.

        :param inputs: Input tensor
        :return: Output tensor
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)

        return x



class GatedMLP(nn.Module):
    """An implementation of a Gated multi-layer perceptron."""

    def __init__(self, in_feats: int, dims: Sequence[int], activate_last: bool = True, use_bias: bool = True):
        """:param in_feats: Dimension of input features.
        :param dims: Architecture of neural networks.
        :param activate_last: Whether applying activation to last layer or not.
        :param use_bias: Whether applying bias in MLP.
        """
        super().__init__()
        self.in_feats = in_feats
        self.dims = [in_feats, *dims]
        self._depth = len(dims)
        self.layers = nn.Sequential()
        self.gates = nn.Sequential()
        self.use_bias = use_bias
        self.activate_last = activate_last
        for i, (in_dim, out_dim) in enumerate(zip(self.dims[:-1], self.dims[1:], strict=False)):
            if i < self._depth - 1:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.layers.append(nn.SiLU())
                self.gates.append(nn.SiLU())
            else:
                self.layers.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                if self.activate_last:
                    self.layers.append(nn.SiLU())
                self.gates.append(nn.Linear(in_dim, out_dim, bias=use_bias))
                self.gates.append(nn.Sigmoid())

    def forward(self, inputs: Tensor):
        return self.layers(inputs) * self.gates(inputs)

