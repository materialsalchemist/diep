"""Elemental reference fitting, energy accounting, and checkpoint compatibility."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Subset
from torch_geometric.data import Batch, Data

from diep.pyg.apps.pes import Potential
from diep.pyg.layers import AtomRef
from diep.pyg.models import DIEP
from diep.pyg.utils.training import PotentialLightningModule
from scripts.train_mp_pes_pyg import _get_element_refs


def _training_subset():
    graphs = [Data(node_type=torch.tensor(types)) for types in ([0], [0, 1], [1, 1], [0, 1], [2])]
    dataset = SimpleNamespace(graphs=graphs, labels={"energies": [-3.0, 1e9, -10.0, -8.0, -999.0]})
    # Non-sequential indices also check that graph compositions and labels stay aligned.
    return Subset(dataset, [3, 0, 2])


def test_references_fit_only_training_structures():
    train = _training_subset()
    refs = _get_element_refs(train, ("Si", "O", "H"))
    torch.testing.assert_close(refs, torch.tensor([-3.0, -5.0, 0.0]))
    train.dataset.labels["energies"][1] = float("nan")
    train.dataset.labels["energies"][4] = 1e15
    torch.testing.assert_close(_get_element_refs(train, ("Si", "O", "H")), refs)
    assert not refs.requires_grad


def test_reference_fit_handles_dependent_compositions():
    ref = AtomRef(max_z=3)
    graphs = [Data(node_type=torch.tensor(types)) for types in ([0, 1], [0, 1, 0, 1])]
    ref.fit(graphs, np.array([-8.0, -16.0]))
    torch.testing.assert_close(ref.property_offset, torch.tensor([-4.0, -4.0, 0.0]))


def test_resume_uses_saved_references_without_refitting(tmp_path):
    refs = torch.tensor([-7.0, -2.0, 0.0])
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"model.element_refs.property_offset": refs}}, path)
    # No usable dataset: resuming must get its baseline only from the checkpoint.
    torch.testing.assert_close(_get_element_refs(None, ("Si", "O", "H"), path), refs)


def test_legacy_resume_does_not_add_an_energy_offset(tmp_path, capsys):
    path = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {}}, path)
    assert _get_element_refs(None, ("Si", "O", "H"), path) is None
    assert "preserving its original energy baseline" in capsys.readouterr().out


@pytest.mark.parametrize("refs", [torch.zeros(2), torch.tensor([0.0, float("nan"), 0.0])])
def test_invalid_checkpoint_references_are_rejected(tmp_path, refs):
    path = tmp_path / "invalid.ckpt"
    torch.save({"state_dict": {"model.element_refs.property_offset": refs}}, path)
    with pytest.raises(ValueError, match="finite and match"):
        _get_element_refs(None, ("Si", "O", "H"), path)


def test_references_shift_energy_once_and_preserve_derivatives(graphs, element_types, tmp_path):
    model = DIEP(element_types=element_types, nblocks=1, is_intensive=False, integral_mode="sum")
    refs = -torch.arange(1, len(element_types) + 1, dtype=torch.float32)
    base = Potential(model=deepcopy(model), data_std=2.5, data_mean=-1.0).eval()
    potential = Potential(model=model, data_std=2.5, data_mean=-1.0, element_refs=refs).eval()
    samples = [graphs[i] for i in (0, 5, 10, 15)]
    batch = Batch.from_data_list([g.clone() for g, _, _ in samples])
    lat = torch.cat([lat for _, lat, _ in samples])
    e_base, f_base, s_base, _ = base(batch.clone(), lat)
    e_ref, f_ref, s_ref, _ = potential(batch.clone(), lat)
    expected_offset = torch.stack([refs[g.node_type].sum() for g, _, _ in samples])
    torch.testing.assert_close(e_ref, e_base + expected_offset)
    torch.testing.assert_close(f_ref, f_base)
    torch.testing.assert_close(s_ref, s_base)
    assert "element_refs.property_offset" in potential.state_dict()

    potential.save(tmp_path / "export")
    restored = Potential.load(tmp_path / "export").eval()
    torch.testing.assert_close(restored.element_refs.property_offset, refs)
    e_restored, f_restored, s_restored, _ = restored(batch.clone(), lat)
    torch.testing.assert_close(e_restored, e_ref)
    torch.testing.assert_close(f_restored, f_ref)
    torch.testing.assert_close(s_restored, s_ref)


def test_lightning_checkpoint_preserves_reference_parameterization(tmp_path):
    elements = ("Si", "O", "H")
    refs = torch.tensor([-3.0, -5.0, 0.0])
    module = PotentialLightningModule(
        model=DIEP(element_types=elements, nblocks=1, is_intensive=False), element_refs=refs,
    )
    checkpoint = {"state_dict": module.state_dict(), "hyper_parameters": dict(module.hparams)}
    path = tmp_path / "model.ckpt"
    torch.save(checkpoint, path)
    restored = PotentialLightningModule(
        model=DIEP(element_types=elements, nblocks=1, is_intensive=False),
        element_refs=_get_element_refs(None, elements, path),
    )
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.testing.assert_close(restored.model.element_refs.property_offset, refs)
    torch.testing.assert_close(checkpoint["hyper_parameters"]["element_refs"], refs)
