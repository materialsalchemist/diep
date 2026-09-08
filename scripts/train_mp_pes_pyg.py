#!/usr/bin/env python
"""Download the MP PES dataset(s) and train DIEP on them, using the PyG backend.

Combines MPF.2021.2.8 and/or MatPES-PBE-2025.2 (see ``scripts/mp_pes_datasets.py`` for
exactly what each dataset is and how it's fetched), builds a DIEPDataset of PyG graphs,
and trains ``diep.pyg.models.DIEP`` on energies/forces/stresses with
``diep.pyg.utils.training.PotentialLightningModule``.

Example (quick smoke run, no download of the large MPF archive):

    python scripts/train_mp_pes_pyg.py --datasets matpes --max-structures 200 \\
        --epochs 1 --batch-size 8 --accelerator cpu
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import random_split

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mp_pes_datasets import load_datasets  # noqa: E402

from diep.config import DEFAULT_ELEMENTS  # noqa: E402
from diep.pyg.graph.converters import Structure2Graph  # noqa: E402
from diep.pyg.graph.data import DIEPDataset, MGLDataLoader  # noqa: E402
from diep.pyg.models import DIEP  # noqa: E402
from diep.pyg.utils.training import PotentialLightningModule, xavier_init  # noqa: E402


def _split(dataset, val_frac: float, test_frac: float, seed: int):
    n = len(dataset)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    structures, labels = load_datasets(
        data_dir=args.data_dir,
        datasets=args.datasets,
        matpes_functional=args.matpes_functional,
        matpes_version=args.matpes_version,
        max_structures=args.max_structures,
        force_limit=args.force_limit,
    )

    element_types = DEFAULT_ELEMENTS
    converter = Structure2Graph(element_types=element_types, cutoff=args.cutoff)
    dataset = DIEPDataset(
        structures=structures,
        labels=labels,
        converter=converter,
        threebody_cutoff=args.threebody_cutoff,
        include_line_graph=True,
        save_cache=True,
        raw_dir=os.path.join(args.data_dir, "processed"),
    )

    train_set, val_set, test_set = _split(dataset, args.val_frac, args.test_frac, args.seed)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_set,
        val_data=val_set,
        test_data=test_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = DIEP(
        element_types=element_types,
        is_intensive=False,
        cutoff=args.cutoff,
        threebody_cutoff=args.threebody_cutoff,
    )
    xavier_init(model)

    forces = torch.cat([dataset[i][3]["forces"] for i in train_set.indices])
    rms_forces = torch.sqrt(torch.mean(torch.sum(forces**2, dim=1)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5, amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader), eta_min=args.lr * 1.0e-2)

    lit_model = PotentialLightningModule(
        model=model,
        data_std=rms_forces,
        optimizer=optimizer,
        scheduler=scheduler,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        stress_weight=args.stress_weight,
        lr=args.lr,
        include_line_graph=True,
    )

    logger = CSVLogger(save_dir=args.save_dir, name="logs")
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="val_Total_Loss", mode="min", filename="{epoch:04d}-best_model"
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[EarlyStopping(monitor="val_Total_Loss", mode="min", patience=args.patience), checkpoint_callback],
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        gradient_clip_val=2.0,
        inference_mode=False,
    )
    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume
    )
    trainer.test(model=lit_model, dataloaders=test_loader)

    model_export_path = os.path.join(args.save_dir, "trained_model")
    os.makedirs(model_export_path, exist_ok=True)
    lit_model.model.save(model_export_path)
    print(f"Saved trained model to {model_export_path}")


def _parse_args():
    parser = argparse.ArgumentParser(description="Train DIEP (PyG) on MP PES data")
    parser.add_argument("--data-dir", default="data/mp_pes")
    parser.add_argument("--datasets", choices=["mpf", "matpes", "both"], default="both")
    parser.add_argument("--matpes-functional", default="PBE")
    parser.add_argument("--matpes-version", default="2025.2")
    parser.add_argument("--max-structures", type=int, default=None, help="per-dataset cap, for smoke runs")
    parser.add_argument("--force-limit", type=float, default=None, help="MPF-only max-force outlier filter")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--threebody-cutoff", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--stress-weight", type=float, default=0.1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="logs/mp_pes_pyg")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint (.ckpt) to resume training from")
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
