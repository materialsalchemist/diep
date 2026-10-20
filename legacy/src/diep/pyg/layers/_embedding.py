"""Embedding block for the PyG backend.

:class:`diep.layers._embedding.EmbeddingBlock` operates purely on tensors -- it never
touches a graph object -- so it is reused verbatim rather than duplicated. Re-exported here
so that ``diep.pyg.layers`` is a complete namespace and importing it does not pull in DGL.
"""

from __future__ import annotations

from diep.layers._embedding import EmbeddingBlock

__all__ = ["EmbeddingBlock"]
