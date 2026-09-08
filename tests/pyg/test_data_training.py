"""Pure-PyG tests for the dataset/dataloader/training pipeline.

No DGL / stub required -- see tests/pyg/conftest.py.
"""

from __future__ import annotations

import numpy as np
import torch
from diep.pyg.graph.compute import assert_lg_invariants
from diep.pyg.graph.converters import Structure2Graph
from diep.pyg.graph.data import DIEPDataset, MGLDataLoader, collate_fn_pes
from diep.pyg.models import DIEP
from diep.pyg.utils.training import PotentialLightningModule
from torch.utils.data import Subset

CUTOFF = 5.0
THREEBODY_CUTOFF = 4.0


def _make_dataset(tmp_path, structures, element_types):
    converter = Structure2Graph(element_types=element_types, cutoff=CUTOFF)
    labels = {
        "energies": [float(-5.0 - 0.1 * i) for i in range(len(structures))],
        "forces": [np.zeros((len(s), 3)).tolist() for s in structures],
        "stresses": [np.zeros((3, 3)).tolist() for _ in structures],
    }
    save_dir = str(tmp_path)
    dataset = DIEPDataset(
        converter=converter,
        threebody_cutoff=THREEBODY_CUTOFF,
        include_line_graph=True,
        structures=structures,
        labels=labels,
        save_dir=save_dir,
        raw_dir=save_dir,
        save_cache=True,
    )
    return dataset, save_dir


def test_dataset_builds_valid_line_graphs(tmp_path, structures, element_types):
    dataset, _ = _make_dataset(tmp_path, structures, element_types)
    assert len(dataset) == len(structures)
    data, _, _, _ = dataset[0]
    assert_lg_invariants(data)


def test_dataset_cache_round_trip(tmp_path, structures, element_types):
    dataset, save_dir = _make_dataset(tmp_path, structures, element_types)
    converter = Structure2Graph(element_types=element_types, cutoff=CUTOFF)
    reloaded = DIEPDataset(
        converter=converter,
        threebody_cutoff=THREEBODY_CUTOFF,
        include_line_graph=True,
        structures=structures,
        labels=dataset.labels,
        save_dir=save_dir,
        raw_dir=save_dir,
    )
    assert len(reloaded) == len(dataset)
    d0, _, _, _ = dataset[0]
    r0, _, _, _ = reloaded[0]
    assert torch.equal(d0.triple_index, r0.triple_index)
    assert torch.equal(d0.edge_index, r0.edge_index)


def test_dataloader_batches_pass_invariants(tmp_path, structures, element_types):
    dataset, _ = _make_dataset(tmp_path, structures, element_types)
    train, val = Subset(dataset, list(range(10))), Subset(dataset, list(range(10, len(dataset))))
    train_loader, val_loader = MGLDataLoader(train, val, collate_fn=collate_fn_pes, batch_size=3, num_workers=0)
    batch = next(iter(train_loader))
    g, lat, state_attr, energies, forces, stresses = batch
    assert_lg_invariants(g)
    assert energies.shape[0] == lat.shape[0]
    assert forces.shape[0] == g.num_nodes
    assert next(iter(val_loader)) is not None


def test_training_step_produces_finite_gradients(tmp_path, structures, element_types):
    dataset, _ = _make_dataset(tmp_path, structures, element_types)
    train_loader, _ = MGLDataLoader(
        Subset(dataset, list(range(10))),
        Subset(dataset, list(range(10, len(dataset)))),
        collate_fn=collate_fn_pes,
        batch_size=3,
        num_workers=0,
    )
    batch = next(iter(train_loader))

    model = DIEP(
        element_types=element_types, nblocks=2, is_intensive=False, cutoff=CUTOFF, threebody_cutoff=THREEBODY_CUTOFF
    )
    module = PotentialLightningModule(model=model, stress_weight=0.01, include_line_graph=True, lr=1e-4)

    results, batch_size = module.step(batch)
    assert batch_size == 3
    assert torch.isfinite(results["Total_Loss"]).all()

    results["Total_Loss"].backward()
    params_with_grad = [p for p in model.parameters() if p.grad is not None]
    assert params_with_grad
    assert all(torch.isfinite(p.grad).all() for p in params_with_grad)
