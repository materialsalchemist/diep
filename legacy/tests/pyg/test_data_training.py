"""Pure-PyG tests for the dataset/dataloader/training pipeline.

No DGL / stub required -- see tests/pyg/conftest.py.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from diep.pyg.graph.compute import assert_lg_invariants
from diep.pyg.graph.converters import Structure2Graph
from diep.pyg.graph.data import DIEPDataset, MGLDataLoader, MaxAtomsBatchSampler, collate_fn_pes
from diep.pyg.models import DIEP
from diep.pyg.utils.training import PotentialLightningModule
from torch.utils.data import Subset

CUTOFF = 5.0
THREEBODY_CUTOFF = 4.0


def test_atom_sampler_ddp_schedule_is_independent_of_global_rng():
    counts = [2, 7, 3, 9, 4, 1, 6, 8] * 7
    samplers = [MaxAtomsBatchSampler(counts, 12, rank=rank, num_replicas=4) for rank in range(4)]
    previous = None
    for epoch in range(3):
        schedules = []
        for rank, sampler in enumerate(samplers):
            torch.manual_seed(100 * epoch + rank)
            torch.rand(17 * rank)
            sampler.set_epoch(epoch)
            batches = list(sampler)
            assert len(batches) == len(sampler)
            assert all(sum(counts[i] for i in batch) <= 12 for batch in batches)
            schedules.append(batches)
        assert len({len(batches) for batches in schedules}) == 1
        interleaved = [batch for step in zip(*schedules) for batch in step]
        n_packs = len(MaxAtomsBatchSampler(counts, 12))
        indices = [i for batch in interleaved[:n_packs] for i in batch]
        assert sorted(indices) == list(range(len(counts)))
        assert schedules != previous
        previous = schedules


def test_atom_sampler_keeps_small_evaluation_splits_on_every_rank():
    for rank in range(4):
        sampler = MaxAtomsBatchSampler([3, 4], 10, shuffle=False, rank=rank, num_replicas=4)
        assert len(sampler) == 1
        assert list(sampler) == [[0, 1]]


def test_atom_sampler_epoch_hook_and_resume():
    from lightning.fabric.utilities.data import _set_sampler_epoch
    from torch.utils.data import DataLoader

    counts = [3, 7] * 20
    sampler = MaxAtomsBatchSampler(counts, 15, rank=2, num_replicas=4)
    loader = DataLoader(counts, batch_sampler=sampler)
    _set_sampler_epoch(loader, 9)
    restored = MaxAtomsBatchSampler(counts, 15, rank=2, num_replicas=4)
    restored.set_epoch(9)
    assert sampler.epoch == 9
    assert list(sampler) == list(restored)


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


@pytest.mark.parametrize("mmap_cache", [False, True])
def test_dataset_cache_round_trip(tmp_path, structures, element_types, mmap_cache):
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
        mmap_cache=mmap_cache,
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

    # Evaluation needs coordinate gradients without leaking grad mode into DDP's
    # post-forward hook (which would wait for a nonexistent validation backward).
    module.eval()
    with torch.no_grad():
        results, _ = module.step(batch)
        assert not torch.is_grad_enabled()
        assert all(torch.isfinite(value).all() for value in results.values())
        assert all(value.device == results["Total_Loss"].device for value in results.values())
        assert all(not value.requires_grad for value in results.values())
