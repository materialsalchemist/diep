"""Regression tests for three defects fixed on 2026-09-29 (third pass).

None of these changes what the model predicts; the first two change what gets *reported*,
which is worse in a way, because a wrong number that never raises gets written into a paper.

1. ``step`` returned ``preds[0].numel()`` -- the *structure* count -- and the mixin passed it
   as ``batch_size`` for every key in one ``log_dict`` call. Lightning uses ``batch_size`` as
   the weight when reducing an ``on_epoch=True`` metric, but ``Force_MAE`` is a mean over
   atoms and ``Stress_MAE`` over 3x3 tensors, so their epoch averages were weighted by the
   wrong quantity. Only heterogeneous batches show it, which is why uniform-cell runs passed.

2. The torchmetrics instances were never ``reset()``. Called as functions they *return* the
   batch value (what Lightning logs, and correct), while also accumulating internal state, so
   ``compute()`` returned a number pooled over every batch since construction.

3. ``_build_bond_frames_vectorized`` normalised the bond vector with ``torch.norm`` followed
   by ``clamp_min`` -- the one place in ``_diep_core`` still using the pattern the rest of the
   file documents as forbidden. ``torch.norm`` at exactly zero has a finite first derivative
   but a NaN second one, and the clamp runs too late to prevent it, so a coincident bond gave
   a NaN Hessian row while its energy and force stayed finite.

Each test was confirmed to fail with its fix reverted.
"""

from __future__ import annotations

import lightning as pl
import pytest
import torch
import torchmetrics

from diep_pyg.layers._diep_core import _build_bond_frames_vectorized
from diep_pyg.models._diep import DIEP
from diep_pyg.utils.training import MatglLightningModuleMixin, PotentialLightningModule


class _WeightProbe(MatglLightningModuleMixin, pl.LightningModule):
    """Minimal module that feeds fixed per-batch metrics through the real logging path.

    Uses the mixin's own ``_log_results`` so the test exercises the reduction Lightning
    actually performs, rather than a reimplementation of it.
    """

    def __init__(self, batches):
        super().__init__()
        self.sync_dist = False
        self.probe_batches = batches
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def step(self, batch):  # noqa: D102
        n_structures, n_atoms, force_mae = batch
        results = {
            "Total_Loss": self.weight.sum() * 0,
            "Energy_MAE": torch.tensor(0.0),
            "Force_MAE": torch.tensor(float(force_mae)),
        }
        weights = {
            "Total_Loss": n_structures,
            "Energy_MAE": n_structures,
            "Force_MAE": n_atoms * 3,
        }
        return results, weights

    def train_dataloader(self):  # noqa: D102
        return torch.utils.data.DataLoader(self.probe_batches, batch_size=None, collate_fn=lambda x: x)

    def configure_optimizers(self):  # noqa: D102
        return torch.optim.SGD(self.parameters(), lr=0.0)


def test_force_metrics_are_weighted_by_atom_count_not_structure_count():
    """The epoch reduction must weight a mean-over-atoms by the number of force components.

    Batch A: 1 structure, 100 atoms, force MAE 1.0.
    Batch B: 10 structures, 10 atoms total, force MAE 0.0.

    Structure-weighted gives 1/11 = 0.0909; atom-weighted gives 100/110 = 0.9091. A 10x gap,
    so this cannot pass by coincidence.
    """
    module = _WeightProbe([(1, 100, 1.0), (10, 10, 0.0)])
    trainer = pl.Trainer(
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
    )
    trainer.fit(module)

    force_mae = float(trainer.callback_metrics["train_Force_MAE"])
    assert force_mae == pytest.approx(100 / 110, abs=1e-4), (
        f"force MAE reduced to {force_mae:.4f}; atom-weighted is {100 / 110:.4f}, "
        f"structure-weighted (the defect) is {1 / 11:.4f}"
    )


def test_loss_fn_reports_a_weight_per_metric():
    """``loss_fn`` must hand back each metric's own denominator, not one shared count."""
    model = DIEP(element_types=("Si",), nblocks=1)
    module = PotentialLightningModule(model=model, stress_weight=0.0)

    n_structures, n_atoms = 4, 10
    labels = (torch.randn(n_structures), torch.randn(n_atoms, 3), torch.zeros(n_structures))
    preds = (torch.randn(n_structures), torch.randn(n_atoms, 3), torch.zeros(1))
    _, weights = module.loss_fn(
        loss=torch.nn.MSELoss(),
        labels=labels,
        preds=preds,
        num_atoms=torch.full((n_structures,), float(n_atoms) / n_structures),
    )

    assert weights["Energy_MAE"] == n_structures
    assert weights["Energy_RMSE"] == n_structures
    # Forces are a mean over every component, not over structures.
    assert weights["Force_MAE"] == n_atoms * 3
    assert weights["Force_RMSE"] == n_atoms * 3
    assert weights["Force_MAE"] != weights["Energy_MAE"], "force and energy share a weight again"


def test_metrics_do_not_pool_across_batches():
    """``compute()`` after a reset must reflect only the batches since that reset.

    Without the reset, two batches of MAE 0.1 and 0.5 pool to 0.3 and keep drifting for the
    lifetime of the module.
    """
    model = DIEP(element_types=("Si",), nblocks=1)
    module = PotentialLightningModule(model=model)

    module.mae(torch.zeros(4), torch.full((4,), 0.1))
    module.mae(torch.zeros(4), torch.full((4,), 0.5))
    assert float(module.mae.compute()) == pytest.approx(0.3, abs=1e-6), "precondition: state pools"

    module._reset_metrics()
    module.mae(torch.zeros(4), torch.full((4,), 0.2))
    assert float(module.mae.compute()) == pytest.approx(0.2, abs=1e-6), (
        "compute() still carries batches from before the reset"
    )


def test_reset_reaches_metrics_stored_as_submodules():
    """The reset must scan ``self.modules()``, not ``vars(self)``.

    Assigning an ``nn.Module`` attribute puts it in ``_modules``, not ``__dict__``, so a
    ``vars()`` scan finds zero metrics and the reset silently does nothing -- a fix that looks
    right and has no effect.
    """
    model = DIEP(element_types=("Si",), nblocks=1)
    module = PotentialLightningModule(model=model)

    assert not [v for v in vars(module).values() if isinstance(v, torchmetrics.Metric)]
    found = [m for m in module.modules() if isinstance(m, torchmetrics.Metric)]
    assert len(found) >= 8, f"expected the eight metric instances, found {len(found)}"

    for metric in found:
        metric(torch.zeros(2), torch.full((2,), 0.5))
    module._reset_metrics()
    for metric in found:
        assert metric.update_count == 0, "a metric survived the reset with state intact"


def test_coincident_bond_does_not_produce_a_nan_hessian():
    """A zero-length bond must give a finite second derivative.

    ``torch.norm`` at zero has a NaN second derivative, and the old ``clamp_min`` ran after
    the norm, too late to prevent it. Energies and forces stayed finite, so the NaN appeared
    only in the Hessian and only for that edge -- no warning anywhere.
    """
    pos_src = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True)
    # Second edge has coincident endpoints: a duplicate site, or a self-image pair that slips
    # past the converter's numerical_tol filter.
    pos_dst = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float64, requires_grad=True)

    _, e_x, _ = _build_bond_frames_vectorized(pos_src, pos_dst)
    assert torch.isfinite(e_x).all()

    first, = torch.autograd.grad(e_x.sum(), pos_src, create_graph=True)
    assert torch.isfinite(first).all(), "first derivative is not finite"
    second, = torch.autograd.grad(first.sum(), pos_src, allow_unused=True)
    assert second is not None
    assert torch.isfinite(second).all(), f"NaN/Inf in second derivative: {second}"


def test_bond_frame_softening_leaves_ordinary_bonds_unchanged():
    """Softening must not perturb a frame built from well-separated atoms.

    The eps**2 floor is 1e-20, so for bonds of ordinary length the unit vectors should agree
    with the unsoftened result to roughly float64 precision.
    """
    torch.manual_seed(0)
    pos_src = torch.randn(32, 3, dtype=torch.float64)
    pos_dst = pos_src + torch.randn(32, 3, dtype=torch.float64) * 0.5 + 2.0

    _, e_x, e_y = _build_bond_frames_vectorized(pos_src, pos_dst)
    reference = (pos_dst - pos_src) / torch.norm(pos_dst - pos_src, dim=1, keepdim=True)

    assert torch.allclose(e_x, reference, atol=1e-12)
    # Frame axes stay orthonormal.
    assert torch.allclose((e_x * e_x).sum(dim=1), torch.ones(32, dtype=torch.float64), atol=1e-12)
    assert torch.allclose((e_y * e_y).sum(dim=1), torch.ones(32, dtype=torch.float64), atol=1e-12)
    assert torch.allclose((e_x * e_y).sum(dim=1), torch.zeros(32, dtype=torch.float64), atol=1e-12)
