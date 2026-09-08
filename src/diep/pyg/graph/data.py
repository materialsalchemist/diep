"""Datasets, collate functions and dataloaders for the PyG backend.

Mirrors :mod:`diep.graph.data`. The one structural difference: the three-body line graph is
not a separate object here -- it is ``triple_index`` / ``n_triple_ij`` stored on the graph --
so there is no ``l_g`` to carry through the collate functions or the training step. The
``include_line_graph`` flag is kept because it still decides whether the line graph is built
at all, and because it keeps call sites recognisable against the DGL API.

Batching is PyG's: ``Batch.from_data_list`` offsets ``edge_index`` by the running node count
and, via :meth:`diep.pyg.graph.compute.DIEPData.__inc__`, offsets ``triple_index`` by the
running *bond* count. That is what keeps the line graph in parent-bond index space across a
batch, exactly as ``dgl.batch`` does for a remapped DGL line graph.
"""

from __future__ import annotations

import json
import os
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch
from tqdm import trange

import diep
from diep.pyg.graph.compute import compute_pair_vector_and_distance, create_line_graph

if TYPE_CHECKING:
    from collections.abc import Callable

    from diep.pyg.graph.converters import GraphConverter


def collate_fn_graph(batch, include_line_graph: bool = False, multiple_values_per_target: bool = False):
    """Merge a list of PyG graphs into a batch.

    Args:
        batch: list of ``(data, lattice, state_attr, labels)`` tuples.
        include_line_graph: accepted for signature parity with the DGL collate; the line
            graph travels on the graph itself, so nothing extra is returned.
        multiple_values_per_target: whether each label holds a vector rather than a scalar.

    Returns:
        (batched graph, lattices, state attributes, labels)
    """
    del include_line_graph
    graphs, lattices, state_attr, labels = map(list, zip(*batch, strict=False))
    g = Batch.from_data_list(graphs)
    labels = (
        torch.vstack([next(iter(d.values())) for d in labels])
        if multiple_values_per_target
        else torch.tensor([next(iter(d.values())) for d in labels], dtype=diep.float_th)
    )
    state_attr = torch.stack(state_attr)
    lat = lattices[0] if len(graphs) == 1 else torch.squeeze(torch.stack(lattices))
    return g, lat, state_attr, labels


def collate_fn_pes(batch, include_stress: bool = True, include_line_graph: bool = False, include_magmom: bool = False):
    """Merge a list of PyG graphs into a batch, for potential (energy/force/stress) training.

    Args:
        batch: list of ``(data, lattice, state_attr, labels)`` tuples.
        include_stress: whether stress labels are present.
        include_line_graph: accepted for signature parity with the DGL collate.
        include_magmom: whether magnetic-moment labels are present.

    Returns:
        (batched graph, lattices, state attributes, energies, forces, stresses[, magmoms])
    """
    del include_line_graph
    graphs, lattices, state_attr, labels = map(list, zip(*batch, strict=False))
    g = Batch.from_data_list(graphs)
    e = torch.tensor([d["energies"] for d in labels], dtype=diep.float_th)
    f = torch.vstack([d["forces"] for d in labels])
    s = (
        torch.vstack([d["stresses"] for d in labels])
        if include_stress
        else torch.tensor(np.zeros(e.size(dim=0)), dtype=diep.float_th)
    )
    m = (
        torch.vstack([d["magmoms"] for d in labels])
        if include_magmom
        else torch.tensor(np.zeros(e.size(dim=0)), dtype=diep.float_th)
    )
    state_attr = torch.stack(state_attr)
    lat = torch.stack(lattices)
    if include_magmom:
        return g, torch.squeeze(lat), state_attr, e, f, s, m
    return g, torch.squeeze(lat), state_attr, e, f, s


def MGLDataLoader(  # noqa: N802 (name matches the DGL API)
    train_data: Subset,
    val_data: Subset,
    collate_fn: Callable | None = None,
    test_data: Subset | None = None,
    **kwargs,
) -> tuple[DataLoader, ...]:
    """Dataloaders for DIEP training on PyG graphs.

    Args:
        train_data: training subset.
        val_data: validation subset.
        collate_fn: collate function; inferred from the available labels when None.
        test_data: optional test subset.
        **kwargs: pass-through to ``torch.utils.data.DataLoader`` (batch_size,
            num_workers, pin_memory, generator, ...). A plain torch DataLoader is used
            rather than ``torch_geometric.loader.DataLoader`` because the dataset yields
            ``(graph, lattice, state_attr, labels)`` tuples; PyG's loader would ignore the
            collate function and batch each tuple element with its own collater.

    Returns:
        Train, validation and (optionally) test dataloaders.
    """
    if collate_fn is None:
        labels = train_data.dataset.labels
        if "forces" not in labels:
            collate_fn = collate_fn_graph
        elif "stresses" not in labels:
            collate_fn = partial(collate_fn_pes, include_stress=False)
        elif "magmoms" not in labels:
            collate_fn = collate_fn_pes
        else:
            collate_fn = partial(collate_fn_pes, include_stress=True, include_magmom=True)

    train_loader = DataLoader(train_data, shuffle=True, collate_fn=collate_fn, **kwargs)
    val_loader = DataLoader(val_data, shuffle=False, collate_fn=collate_fn, **kwargs)
    if test_data is not None:
        return train_loader, val_loader, DataLoader(test_data, shuffle=False, collate_fn=collate_fn, **kwargs)
    return train_loader, val_loader


class DIEPDataset(Dataset):
    """An in-memory dataset of PyG graphs built from pymatgen structures.

    The PyG counterpart of :class:`diep.graph.data.MGLDataset`. Caching is a single
    ``torch.save`` of the graph list rather than DGL's ``save_graphs``.
    """

    def __init__(
        self,
        filename: str = "pyg_graph.pt",
        filename_lattice: str = "lattice.pt",
        filename_state_attr: str = "state_attr.pt",
        filename_labels: str = "labels.json",
        include_line_graph: bool = False,
        converter: GraphConverter | None = None,
        threebody_cutoff: float | None = None,
        structures: list | None = None,
        labels: dict[str, list] | None = None,
        directory_name: str = "DIEPDataset",
        graph_labels: list[int | float] | None = None,
        clear_processed: bool = False,
        save_cache: bool = True,
        raw_dir: str = "./",
        save_dir: str | None = None,
    ):
        """
        Args:
            filename: file name for the stored graphs.
            filename_lattice: file name for the stored lattice matrices.
            filename_state_attr: file name for the stored state attributes.
            filename_labels: file name for the stored labels.
            include_line_graph: whether to build the three-body line graph.
            converter: graph converter (see :mod:`diep.pyg.graph.converters`).
            threebody_cutoff: cutoff for three-body interactions.
            structures: pymatgen structures.
            labels: targets, as a dict of {name: list of values}.
            directory_name: name of the generated directory storing the dataset.
            graph_labels: state attributes.
            clear_processed: drop the structures from memory after conversion.
            save_cache: whether to save the processed dataset.
            raw_dir: directory holding or receiving the input data.
            save_dir: directory to save the processed dataset. Defaults to raw_dir.
        """
        self.filename = filename
        self.filename_lattice = filename_lattice
        self.filename_state_attr = filename_state_attr
        self.filename_labels = filename_labels
        self.include_line_graph = include_line_graph
        self.converter = converter
        self.structures = structures or []
        self.labels = labels or {}
        for k, v in self.labels.items():
            self.labels[k] = v.tolist() if isinstance(v, np.ndarray) else v
        self.threebody_cutoff = threebody_cutoff
        self.graph_labels = graph_labels
        self.clear_processed = clear_processed
        self.save_cache = save_cache
        self.save_path = os.path.join(save_dir if save_dir is not None else raw_dir, directory_name)

        if self.has_cache():
            print(f"Warning! Loading graphs from processed cache at {self.save_path}.")
            self.load()
        else:
            print("Cache not found, processing graphs...")
            self.process()
            self.save()

    def has_cache(self) -> bool:
        """Whether a processed cache already exists at ``save_path``."""
        files = [self.filename, self.filename_lattice, self.filename_state_attr, self.filename_labels]
        return all(os.path.exists(os.path.join(self.save_path, f)) for f in files)

    def process(self):
        """Convert the pymatgen structures into PyG graphs."""
        graphs, lattices, state_attrs = [], [], []
        for idx in trange(len(self.structures)):
            structure = self.structures[idx]
            graph, lattice, state_attr = self.converter.get_graph(structure)
            graph.pos = torch.tensor(structure.cart_coords, dtype=diep.float_th)
            graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
            bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
            graph.bond_vec = bond_vec
            graph.bond_dist = bond_dist
            if self.include_line_graph:
                create_line_graph(graph, self.threebody_cutoff)
            # positions and offshifts are recomputed at forward time from frac_coords and
            # the (possibly strained) lattice, so they are not part of the cached graph
            del graph.pos, graph.pbc_offshift, graph.bond_vec, graph.bond_dist
            graphs.append(graph)
            lattices.append(lattice)
            state_attrs.append(state_attr)

        state_attrs = (
            torch.tensor(self.graph_labels).long()
            if self.graph_labels is not None
            else torch.tensor(np.array(state_attrs), dtype=diep.float_th)
        )
        if self.clear_processed:
            del self.structures
            self.structures = []

        self.graphs = graphs
        self.lattices = lattices
        self.state_attr = state_attrs
        return self.graphs, self.lattices, self.state_attr

    def save(self):
        """Save the processed graphs and labels to ``save_path``."""
        if self.save_cache is False:
            return
        os.makedirs(self.save_path, exist_ok=True)
        if self.labels:
            with open(os.path.join(self.save_path, self.filename_labels), "w") as file:
                json.dump(self.labels, file)
        else:
            with open(os.path.join(self.save_path, self.filename_labels), "w") as file:
                json.dump({}, file)
        torch.save(self.graphs, os.path.join(self.save_path, self.filename))
        torch.save(self.lattices, os.path.join(self.save_path, self.filename_lattice))
        torch.save(self.state_attr, os.path.join(self.save_path, self.filename_state_attr))

    def load(self):
        """Load processed graphs from ``save_path``."""
        self.graphs = torch.load(os.path.join(self.save_path, self.filename), weights_only=False)
        self.lattices = torch.load(os.path.join(self.save_path, self.filename_lattice), weights_only=False)
        self.state_attr = torch.load(os.path.join(self.save_path, self.filename_state_attr), weights_only=False)
        with open(os.path.join(self.save_path, self.filename_labels)) as f:
            self.labels = json.load(f)

    def __getitem__(self, idx: int):
        """Return ``(graph, lattice, state_attr, labels)`` for one structure."""
        return (
            self.graphs[idx],
            self.lattices[idx],
            self.state_attr[idx],
            {
                k: torch.tensor(v[idx], dtype=diep.float_th)
                for k, v in self.labels.items()
                if not isinstance(v[idx], str)
            },
        )

    def __len__(self):
        """Number of structures in the dataset."""
        return len(self.graphs)
