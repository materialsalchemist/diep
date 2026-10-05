"""Lightning modules for training DIEP on PyG graphs.

Translation of :mod:`diep.utils.training`. Two differences from the DGL version, both
consequences of the PyG data model:

* There is no ``l_g`` in the batch -- the three-body line graph is carried on the graph as
  ``triple_index`` / ``n_triple_ij``, so batches are one element shorter.
* ``PotentialLightningModule`` forwards ``use_edges`` to :class:`Potential`. The DGL version
  wraps the model in a ``Potential`` that defaults ``use_edges=False`` and applies it
  unconditionally, which silently switches the whole triplet path off during training.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from diep_pyg.apps.pes import Potential

if TYPE_CHECKING:
    import numpy as np
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


class MatglLightningModuleMixin:
    """Mix-in class implementing common functions for training."""

    def _log_results(self, results: dict, weights: dict, stage: str) -> None:
        """Log every metric under ``stage``, each weighted by its own element count.

        Lightning reduces an ``on_epoch=True`` metric as a ``batch_size``-weighted mean, so
        the weight has to match what the metric averaged over. A single ``log_dict`` call
        takes one ``batch_size`` for every key, which forced the force metrics (means over
        atoms) and the stress metrics (means over 3x3 tensors) to be weighted by the
        structure count. Grouping the keys by weight and issuing one call per group is the
        smallest change that gives each metric its correct denominator; the alternative, a
        separate ``self.log`` per key, loses ``log_dict``'s single-call semantics for no gain.
        """
        by_weight: dict[int, dict] = {}
        for key, value in results.items():
            by_weight.setdefault(int(weights[key]), {})[f"{stage}_{key}"] = value
        for weight, group in by_weight.items():
            self.log_dict(
                group,
                # A zero-element metric must not be reduced with weight 0: Lightning would
                # divide by a zero total if every batch in the epoch were empty. Weight 1 and
                # a placeholder value of 0 is the pre-existing behaviour for absent targets.
                batch_size=max(weight, 1),
                on_epoch=True,
                on_step=False,
                prog_bar=True,
                sync_dist=self.sync_dist,
            )

    def _reset_metrics(self) -> None:
        """Clear the accumulated state of every torchmetrics instance on this module.

        The metrics are called as functions, so each call *returns* that batch's value --
        which is what gets logged, and is correct -- but also folds the batch into internal
        running state that was never cleared. `compute()` therefore returned a number pooled
        over every batch of every epoch since construction (batches of MAE 0.1 and 0.5 give
        0.3, then keep drifting), and the state grew without bound. Nothing in this module
        calls `compute()`, which is why the logged scalars looked right and hid it; any
        callback, epoch-level aggregation or checkpoint metric that does read the state got a
        meaningless value.

        Reset at every epoch boundary, and for each stage, because the same instances are
        shared by train, validation and test: without that, validation would pool the
        training batches that preceded it in the same epoch.
        """
        # `self.modules()`, not `vars(self)`: assigning an `nn.Module` attribute stores it in
        # `_modules` rather than `__dict__`, so a `vars()` scan finds no metrics at all and
        # the reset is a silent no-op.
        for module in self.modules():
            if isinstance(module, torchmetrics.Metric):
                module.reset()

    def on_train_epoch_start(self) -> None:
        """Clear pooled metric state before the epoch's first training batch."""
        self._reset_metrics()

    def on_validation_epoch_start(self) -> None:
        """Clear pooled metric state before validating, so training batches do not leak in."""
        self._reset_metrics()

    def on_test_epoch_start(self) -> None:
        """Clear pooled metric state before testing."""
        self._reset_metrics()

    def training_step(self, batch: tuple, batch_idx: int):
        """Run one training step and log its metrics."""
        results, weights = self.step(batch)
        self._log_results(results, weights, "train")
        return results["Total_Loss"]

    def validation_step(self, batch: tuple, batch_idx: int):
        """Run one validation step and log its metrics."""
        results, weights = self.step(batch)
        self._log_results(results, weights, "val")
        return results["Total_Loss"]

    def test_step(self, batch: tuple, batch_idx: int):
        """Run one test step and log its metrics."""
        results, weights = self.step(batch)
        self._log_results(results, weights, "test")
        return results

    def configure_optimizers(self):
        """Configure the optimizer and learning-rate scheduler."""
        optimizer = (
            torch.optim.Adam(self.parameters(), lr=self.lr, eps=1e-8)
            if self.optimizer is None
            else self.optimizer
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.decay_steps, eta_min=self.lr * self.decay_alpha
            )
            if self.scheduler is None
            else self.scheduler
        )
        return [optimizer], [scheduler]

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """Run one prediction step."""
        return self.step(batch)


class PotentialLightningModule(MatglLightningModuleMixin, pl.LightningModule):
    """A LightningModule for training DIEP potentials on energies, forces and stresses."""

    def __init__(
        self,
        model,
        element_refs: np.ndarray | None = None,
        include_line_graph: bool = True,
        energy_weight: float = 1.0,
        force_weight: float = 1.0,
        stress_weight: float = 0.0,
        magmom_weight: float = 0.0,
        data_mean: float = 0.0,
        data_std: float = 1.0,
        loss: str = "mse_loss",
        loss_params: dict | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        lr: float = 0.001,
        decay_steps: int = 1000,
        decay_alpha: float = 0.01,
        sync_dist: bool = False,
        allow_missing_labels: bool = False,
        magmom_target: Literal["absolute", "symbreak"] | None = "absolute",
        use_edges: bool | None = None,
        **kwargs,
    ):
        """
        Args:
            model: the DIEP model to train.
            element_refs: element offsets for the PES.
            include_line_graph: whether the three-body channel is used. Unlike the DGL
                version this defaults to True and is forwarded to the model, so three-body
                interactions are not silently disabled.
            energy_weight: relative importance of energy.
            force_weight: relative importance of forces.
            stress_weight: relative importance of stress.
            magmom_weight: relative importance of magmom predictions.
            data_mean: mean of the training data.
            data_std: standard deviation of the training data.
            loss: "mse_loss", "huber_loss", "smooth_l1_loss" or "l1_loss".
            loss_params: extra parameters for the loss function.
            optimizer: optimizer for training.
            scheduler: learning-rate scheduler.
            lr: learning rate.
            decay_steps: number of steps for the learning-rate decay.
            decay_alpha: sets the minimum learning rate.
            sync_dist: whether to sync logging across GPU workers.
            allow_missing_labels: allow NaN labels, skipped in the loss.
            magmom_target: "absolute", "symbreak", or None.
            use_edges: overrides the model's triplet usage. Defaults to
                ``include_line_graph``.
            **kwargs: passthrough to the parent init.
        """
        assert energy_weight >= 0, f"energy_weight has to be >=0. Got {energy_weight}!"
        assert force_weight >= 0, f"force_weight has to be >=0. Got {force_weight}!"
        assert stress_weight >= 0, f"stress_weight has to be >=0. Got {stress_weight}!"
        assert magmom_weight >= 0, f"magmom_weight has to be >=0. Got {magmom_weight}!"
        # A non-zero magmom_weight builds a Potential with calc_magmom=True, which this
        # package cannot serve -- the DIEP model has no site-property head. Caught here so the
        # failure names the weight that caused it, rather than surfacing from Potential during
        # the first training step.
        if magmom_weight != 0:
            raise NotImplementedError(
                "magmom_weight must be 0: the DIEP model has no site-property head, so magmoms "
                "cannot be predicted or trained against. See Potential(calc_magmom=...)."
            )

        super().__init__(**kwargs)

        # One metric instance per quantity. torchmetrics objects accumulate state across
        # calls, so a single shared `self.mae` would pool energies, forces, stresses and
        # magmoms -- incommensurate quantities -- into one running average. Lightning logs
        # the per-batch value each call returns, which hides the mixing, but `.compute()`
        # (and anything that reads metric state, e.g. epoch-level aggregation) would be
        # meaningless. Registered as attributes so Lightning moves them with the module.
        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.force_mae = torchmetrics.MeanAbsoluteError()
        self.force_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.stress_mae = torchmetrics.MeanAbsoluteError()
        self.stress_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.magmom_mae = torchmetrics.MeanAbsoluteError()
        self.magmom_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.register_buffer("data_mean", torch.as_tensor(data_mean).detach().clone())
        self.register_buffer("data_std", torch.as_tensor(data_std).detach().clone())

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.stress_weight = stress_weight
        self.magmom_weight = magmom_weight
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_alpha = decay_alpha
        self.include_line_graph = include_line_graph

        self.model = Potential(
            model=model,
            element_refs=element_refs,
            calc_stresses=stress_weight != 0,
            calc_magmom=magmom_weight != 0,
            data_std=self.data_std,
            data_mean=self.data_mean,
            use_edges=include_line_graph if use_edges is None else use_edges,
        )
        if loss == "mse_loss":
            self.loss = F.mse_loss
        elif loss == "huber_loss":
            self.loss = F.huber_loss
        elif loss == "smooth_l1_loss":
            self.loss = F.smooth_l1_loss
        else:
            self.loss = F.l1_loss
        self.loss_params = loss_params if loss_params is not None else {}
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.sync_dist = sync_dist
        self.allow_missing_labels = allow_missing_labels
        self.magmom_target = magmom_target
        self.save_hyperparameters(ignore=["model", "optimizer", "scheduler"])

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        """Backfill state-dict keys added since the checkpoint was written."""
        for key in self.state_dict():
            if key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]

    def forward(self, g, lat: torch.Tensor, state_attr: torch.Tensor | None = None):
        """Predict energy, forces, stress, hessian and optionally site-wise properties."""
        if self.model.calc_magmom:
            return self.model(g=g, lat=lat, state_attr=state_attr)
        e, f, s, h = self.model(g=g, lat=lat, state_attr=state_attr)
        return e, f, s, h

    @torch.enable_grad()
    def step(self, batch: tuple):
        """Run the model on one batch and compute the losses.

        Args:
            batch: ``(g, lat, state_attr, energies, forces, stresses[, magmoms])``.

        Returns:
            (results dict, batch size)
        """
        # Forces need autograd even in evaluation. The decorator restores the caller's
        # grad mode before DDP's post-forward hook, so validation cannot accidentally
        # arm a parameter backward pass that will never happen.
        if self.model.calc_magmom:
            g, lat, state_attr, energies, forces, stresses, magmoms = batch
            e, f, s, _, m = self(g=g, lat=lat, state_attr=state_attr)
            preds, labels = (e, f, s, m), (energies, forces, stresses, magmoms)
        else:
            g, lat, state_attr, energies, forces, stresses = batch
            e, f, s, _ = self(g=g, lat=lat, state_attr=state_attr)
            preds, labels = (e, f, s), (energies, forces, stresses)

        if not self.training:
            preds = tuple(pred.detach() for pred in preds)
        num_atoms = self.model._num_nodes_per_graph(g)
        results, weights = self.loss_fn(labels=labels, preds=preds, num_atoms=num_atoms)
        return results, weights

    def loss_fn(self, labels: tuple, preds: tuple, num_atoms: torch.Tensor | None = None, loss: nn.Module | None = None):
        """Compute energy/force/stress/magmom losses and metrics.

        Every target uses the same loss function -- ``loss`` when given, otherwise
        ``self.loss``. Previously energy and force were hard-wired to ``self.loss`` while
        stress and magmom used the ``loss`` argument, so passing a ``loss`` different from
        ``self.loss`` silently optimised a split objective.

        Args:
            labels: ground-truth (energy, force, stress[, magmom]).
            preds: predicted (energy, force, stress[, magmom]).
            num_atoms: atom count of each structure, used to make the energy loss intensive.
            loss: loss function to use for every target. Defaults to ``self.loss``.

        Returns:
            A dict of Total_Loss and per-target MAE / RMSE.
        """
        loss = self.loss if loss is None else loss
        if num_atoms is None:
            num_atoms = torch.ones_like(preds[0])
        if self.allow_missing_labels:
            valid_labels, valid_preds = [], []
            valid_num_atoms = num_atoms
            for index, label in enumerate(labels):
                pred = preds[index]
                if index == 0 and pred.shape == torch.Size([]):
                    pred = pred.view(1)
                # Mask only a target whose prediction is really shaped like its label. A
                # disabled target has a placeholder prediction -- `Potential.forward` returns
                # `torch.zeros(1)` for stress when `calc_stresses=False` -- while the collate
                # function still supplies a full-length label, so applying the label's mask to
                # it raised `IndexError: The shape of the mask [B] ... does not match the shape
                # of the indexed tensor [1]`. That made `allow_missing_labels=True` unusable in
                # exactly the energy-and-forces configuration it is most wanted for. The
                # per-target gates below skip these entries anyway, so passing them through
                # unmasked changes nothing that is read.
                if pred.shape == label.shape:
                    valid = ~torch.isnan(label)
                    valid_labels.append(label[valid])
                    valid_preds.append(pred[valid])
                    if index == 0:
                        valid_num_atoms = num_atoms[valid]
                else:
                    valid_labels.append(label)
                    valid_preds.append(pred)
        else:
            valid_labels, valid_preds = list(labels), list(preds)
            valid_num_atoms = num_atoms

        e_loss = loss(valid_labels[0] / valid_num_atoms, valid_preds[0] / valid_num_atoms, **self.loss_params)
        f_loss = loss(valid_labels[1], valid_preds[1], **self.loss_params)
        e_mae = self.mae(valid_labels[0] / valid_num_atoms, valid_preds[0] / valid_num_atoms)
        f_mae = self.force_mae(valid_labels[1], valid_preds[1])
        e_rmse = self.rmse(valid_labels[0] / valid_num_atoms, valid_preds[0] / valid_num_atoms)
        f_rmse = self.force_rmse(valid_labels[1], valid_preds[1])

        s_mae = s_rmse = m_mae = m_rmse = preds[0].new_zeros(())
        total_loss = self.energy_weight * e_loss + self.force_weight * f_loss

        if self.model.calc_stresses:
            s_loss = loss(valid_labels[2], valid_preds[2], **self.loss_params)
            s_mae = self.stress_mae(valid_labels[2], valid_preds[2])
            s_rmse = self.stress_rmse(valid_labels[2], valid_preds[2])
            total_loss = total_loss + self.stress_weight * s_loss

        if self.model.calc_magmom and labels[3].numel() > 0:
            if self.magmom_target == "symbreak":
                m_loss = torch.min(
                    loss(valid_labels[3], valid_preds[3], **self.loss_params),
                    loss(valid_labels[3], -valid_preds[3], **self.loss_params),
                )
                m_mae = torch.min(
                    self.magmom_mae(valid_labels[3], valid_preds[3]),
                    self.magmom_mae(valid_labels[3], -valid_preds[3]),
                )
                m_rmse = torch.min(
                    self.magmom_rmse(valid_labels[3], valid_preds[3]),
                    self.magmom_rmse(valid_labels[3], -valid_preds[3]),
                )
            else:
                labels_3 = torch.abs(valid_labels[3]) if self.magmom_target == "absolute" else valid_labels[3]
                m_loss = loss(labels_3, valid_preds[3], **self.loss_params)
                m_mae = self.magmom_mae(labels_3, valid_preds[3])
                m_rmse = self.magmom_rmse(labels_3, valid_preds[3])
            total_loss = total_loss + self.magmom_weight * m_loss

        results = {
            "Total_Loss": total_loss,
            "Energy_MAE": e_mae,
            "Force_MAE": f_mae,
            "Stress_MAE": s_mae,
            "Magmom_MAE": m_mae,
            "Energy_RMSE": e_rmse,
            "Force_RMSE": f_rmse,
            "Stress_RMSE": s_rmse,
            "Magmom_RMSE": m_rmse,
        }

        # How many values each metric averaged over. Lightning uses the `batch_size` passed
        # to `log_dict` as the weight when reducing `on_epoch=True`, so a metric must be
        # logged with *its own* denominator. Every key used to be logged with the structure
        # count, but the force metrics are means over atoms (3 components each) and the
        # stress metrics over 3x3 tensors, so their epoch averages came out weighted by the
        # wrong quantity: a batch of 1 structure / 100 atoms at force MAE 1.0 reduced
        # against 10 structures / 10 atoms at 0.0 logged 0.0909 where the atom-weighted
        # value is 0.9091 -- a 10x misreport, and only on heterogeneous batches, which is
        # why uniform-cell runs never showed it.
        #
        # Counted from the *valid* tensors so `allow_missing_labels` masking is reflected.
        # `Total_Loss` is a weighted sum of incommensurate terms with no natural element
        # count; it keeps the structure count, which is the conventional choice and what
        # every previous run used.
        n_structures = int(valid_labels[0].numel())
        weights = {
            "Total_Loss": n_structures,
            "Energy_MAE": n_structures,
            "Energy_RMSE": n_structures,
            "Force_MAE": int(valid_labels[1].numel()),
            "Force_RMSE": int(valid_labels[1].numel()),
            "Stress_MAE": int(valid_labels[2].numel()) if self.model.calc_stresses else n_structures,
            "Stress_RMSE": int(valid_labels[2].numel()) if self.model.calc_stresses else n_structures,
            "Magmom_MAE": n_structures,
            "Magmom_RMSE": n_structures,
        }
        if self.model.calc_magmom and labels[3].numel() > 0:
            weights["Magmom_MAE"] = weights["Magmom_RMSE"] = int(valid_labels[3].numel())
        # A metric computed over an empty selection carries no information; weighting it 0
        # keeps it out of the epoch average instead of dragging it toward the placeholder 0.
        weights = {key: max(value, 0) for key, value in weights.items()}
        return results, weights


def xavier_init(model: nn.Module, gain: float = 1.0, distribution: Literal["uniform", "normal"] = "uniform") -> None:
    """Xavier-initialise every Linear layer of a model.

    Args:
        model: the model to initialise.
        gain: gain factor.
        distribution: "uniform" or "normal".
    """
    if distribution == "uniform":
        init_fn = nn.init.xavier_uniform_
    elif distribution == "normal":
        init_fn = nn.init.xavier_normal_
    else:
        raise ValueError(f"Invalid distribution: {distribution}")

    for param in model.parameters():
        if param.dim() < 2:
            nn.init.zeros_(param)
        else:
            init_fn(param, gain=gain)
