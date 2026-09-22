"""Elemental reference lookup, fitting helper, energy accounting, and checkpoint compatibility."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

from diep.pyg.apps.pes import Potential
from diep.pyg.layers import AtomRef
from diep.pyg.models import DIEP
from diep.pyg.utils.training import PotentialLightningModule
from scripts.train_mp_pes_pyg import _get_element_refs


def test_references_are_looked_up_from_isolated_atom_energies():
    isolated_energies = {"Si": -3.0, "O": -5.0}
    refs = _get_element_refs(("Si", "O", "H"), isolated_energies)
    torch.testing.assert_close(refs, torch.tensor([-3.0, -5.0, 0.0]))
    assert not refs.requires_grad


def test_references_default_to_zero_for_missing_elements(capsys):
    refs = _get_element_refs(("Si", "O", "H"), {"Si": -3.0})
    torch.testing.assert_close(refs, torch.tensor([-3.0, 0.0, 0.0]))
    assert "No isolated-atom reference energy for 2 element(s)" in capsys.readouterr().out


def test_reference_fit_handles_dependent_compositions():
    ref = AtomRef(max_z=3)
    graphs = [Data(node_type=torch.tensor(types)) for types in ([0, 1], [0, 1, 0, 1])]
    ref.fit(graphs, np.array([-8.0, -16.0]))
    torch.testing.assert_close(ref.property_offset, torch.tensor([-4.0, -4.0, 0.0]))


def test_resume_uses_saved_references_without_refitting(tmp_path):
    refs = torch.tensor([-7.0, -2.0, 0.0])
    path = tmp_path / "model.ckpt"
    torch.save({"state_dict": {"model.element_refs.property_offset": refs}}, path)
    # No isolated-atom energies needed: resuming must get its baseline only from the checkpoint.
    torch.testing.assert_close(_get_element_refs(("Si", "O", "H"), None, path), refs)


def test_legacy_resume_does_not_add_an_energy_offset(tmp_path, capsys):
    path = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {}}, path)
    assert _get_element_refs(("Si", "O", "H"), None, path) is None
    assert "preserving its original energy baseline" in capsys.readouterr().out


@pytest.mark.parametrize("refs", [torch.zeros(2), torch.tensor([0.0, float("nan"), 0.0])])
def test_invalid_checkpoint_references_are_rejected(tmp_path, refs):
    path = tmp_path / "invalid.ckpt"
    torch.save({"state_dict": {"model.element_refs.property_offset": refs}}, path)
    with pytest.raises(ValueError, match="finite and match"):
        _get_element_refs(("Si", "O", "H"), None, path)


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
        element_refs=_get_element_refs(elements, None, path),
    )
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.testing.assert_close(restored.model.element_refs.property_offset, refs)
    torch.testing.assert_close(checkpoint["hyper_parameters"]["element_refs"], refs)
