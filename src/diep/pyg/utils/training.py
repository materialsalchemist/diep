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

from diep.pyg.apps.pes import Potential

if TYPE_CHECKING:
    import numpy as np
    from torch.optim import Optimizer
    from torch.optim.lr_scheduler import LRScheduler


class MatglLightningModuleMixin:
    """Mix-in class implementing common functions for training."""

    def training_step(self, batch: tuple, batch_idx: int):
        """Run one training step and log its metrics."""
        results, batch_size = self.step(batch)
        self.log_dict(
            {f"train_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        return results["Total_Loss"]

    def on_train_epoch_end(self):
        """Step the scheduler every epoch."""
        sch = self.lr_schedulers()
        sch.step()

    def validation_step(self, batch: tuple, batch_idx: int):
        """Run one validation step and log its metrics."""
        results, batch_size = self.step(batch)
        self.log_dict(
            {f"val_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
        return results["Total_Loss"]

    def test_step(self, batch: tuple, batch_idx: int):
        """Run one test step and log its metrics."""
        torch.set_grad_enabled(True)
        results, batch_size = self.step(batch)
        self.log_dict(
            {f"test_{key}": val for key, val in results.items()},
            batch_size=batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=self.sync_dist,
        )
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

    def on_test_model_eval(self, *args, **kwargs):
        """Enable gradients during test, which the force calculation needs."""
        super().on_test_model_eval(*args, **kwargs)
        torch.set_grad_enabled(True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        """Run one prediction step."""
        torch.set_grad_enabled(True)
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

        super().__init__(**kwargs)

        self.mae = torchmetrics.MeanAbsoluteError()
        self.rmse = torchmetrics.MeanSquaredError(squared=False)
        self.register_buffer("data_mean", torch.tensor(data_mean))
        self.register_buffer("data_std", torch.tensor(data_std))

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
        self.save_hyperparameters(ignore=["model"])

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

    def step(self, batch: tuple):
        """Run the model on one batch and compute the losses.

        Args:
            batch: ``(g, lat, state_attr, energies, forces, stresses[, magmoms])``.

        Returns:
            (results dict, batch size)
        """
        torch.set_grad_enabled(True)
        if self.model.calc_magmom:
            g, lat, state_attr, energies, forces, stresses, magmoms = batch
            e, f, s, _, m = self(g=g, lat=lat, state_attr=state_attr)
            preds, labels = (e, f, s, m), (energies, forces, stresses, magmoms)
        else:
            g, lat, state_attr, energies, forces, stresses = batch
            e, f, s, _ = self(g=g, lat=lat, state_attr=state_attr)
            preds, labels = (e, f, s), (energies, forces, stresses)

        num_atoms = self.model._num_nodes_per_graph(g)
        results = self.loss_fn(loss=self.loss, preds=preds, labels=labels, num_atoms=num_atoms)
        return results, preds[0].numel()

    def loss_fn(self, loss: nn.Module, labels: tuple, preds: tuple, num_atoms: torch.Tensor | None = None):
        """Compute energy/force/stress/magmom losses and metrics.

        Args:
            loss: loss function.
            labels: ground-truth (energy, force, stress[, magmom]).
            preds: predicted (energy, force, stress[, magmom]).
            num_atoms: atom count of each structure, used to make the energy loss intensive.

        Returns:
            A dict of Total_Loss and per-target MAE / RMSE.
        """
        if num_atoms is None:
            num_atoms = torch.ones_like(preds[0])
        if self.allow_missing_labels:
            valid_labels, valid_preds = [], []
            valid_num_atoms = num_atoms
            for index, label in enumerate(labels):
                valid = ~torch.isnan(label)
                valid_labels.append(label[valid])
                if index == 0:
                    valid_num_atoms = num_atoms[valid]
                    pred = preds[index].view(1) if preds[index].shape == torch.Size([]) else preds[index]
                else:
                    pred = preds[index]
                valid_preds.append(pred[valid])
        else:
            valid_labels, valid_preds = list(labels), list(preds)
            valid_num_atoms = num_atoms

        e_loss = self.loss(valid_labels[0] / valid_num_atoms, valid_preds[0] / valid_num_atoms, **self.loss_params)
        f_loss = self.loss(valid_labels[1], valid_preds[1], **self.loss_params)
        e_mae = self.mae(valid_labels[0] / valid_num_atoms, valid_preds[0] / valid_num_atoms)
        f_mae = self.mae(valid_labels[1], valid_preds[1])
        e_rmse = self.rmse(valid_labels[0] / valid_num_atoms, valid_preds[0] / valid_num_atoms)
        f_rmse = self.rmse(valid_labels[1], valid_preds[1])

        s_mae = s_rmse = m_mae = m_rmse = torch.zeros(1)
        total_loss = self.energy_weight * e_loss + self.force_weight * f_loss

        if self.model.calc_stresses:
            s_loss = loss(valid_labels[2], valid_preds[2], **self.loss_params)
            s_mae = self.mae(valid_labels[2], valid_preds[2])
            s_rmse = self.rmse(valid_labels[2], valid_preds[2])
            total_loss = total_loss + self.stress_weight * s_loss

        if self.model.calc_magmom and labels[3].numel() > 0:
            if self.magmom_target == "symbreak":
                m_loss = torch.min(
                    loss(valid_labels[3], valid_preds[3], **self.loss_params),
                    loss(valid_labels[3], -valid_preds[3], **self.loss_params),
                )
                m_mae = torch.min(self.mae(valid_labels[3], valid_preds[3]), self.mae(valid_labels[3], -valid_preds[3]))
                m_rmse = torch.min(
                    self.rmse(valid_labels[3], valid_preds[3]), self.rmse(valid_labels[3], -valid_preds[3])
                )
            else:
                labels_3 = torch.abs(valid_labels[3]) if self.magmom_target == "absolute" else valid_labels[3]
                m_loss = loss(labels_3, valid_preds[3], **self.loss_params)
                m_mae = self.mae(labels_3, valid_preds[3])
                m_rmse = self.rmse(labels_3, valid_preds[3])
            total_loss = total_loss + self.magmom_weight * m_loss

        return {
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
