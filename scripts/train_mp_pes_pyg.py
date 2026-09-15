#!/usr/bin/env python
"""Download the MP PES dataset(s) and train DIEP on them, using the PyG backend.

Combines MPF.2021.2.8 and/or MatPES-PBE-2025.2 (see ``scripts/mp_pes_datasets.py`` for
exactly what each dataset is and how it's fetched), builds a DIEPDataset of PyG graphs,
and trains ``diep.pyg.models.DIEP`` on energies/forces/stresses with
``diep.pyg.utils.training.PotentialLightningModule``.

Fresh runs fit elemental energy references on the training split. The potential adds
these offsets to the learned residual energy, keeping labels in their original units.
Resumes preserve the checkpoint's reference energies, including legacy runs without
references. Start a new output directory to train a legacy model with fitted references.

Example (quick smoke run, no download of the large MPF archive):

    python scripts/train_mp_pes_pyg.py --datasets matpes --max-structures 200 \\
        --epochs 1 --batch-size 8 --accelerator cpu
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
from diep.pyg.layers import AtomRef  # noqa: E402
from diep.pyg.models import DIEP  # noqa: E402
from diep.pyg.utils.training import PotentialLightningModule, xavier_init  # noqa: E402


class TrainingProgress(pl.Callback):
    """Keep redirected tmux logs readable while a long epoch is running."""

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.is_global_zero and (batch_idx + 1) % 100 == 0:
            print(f"Epoch {trainer.current_epoch}: train batch {batch_idx + 1}/{trainer.num_training_batches}, "
                  f"step {trainer.global_step}, loss {float(outputs['loss']):.6g}", flush=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.is_global_zero and (batch_idx + 1) % 10 == 0:
            print(f"Epoch {trainer.current_epoch}: validation batch {batch_idx + 1}", flush=True)


def _split(dataset, val_frac: float, test_frac: float, seed: int):
    n = len(dataset)
    n_val = int(round(n * val_frac))
    n_test = int(round(n * test_frac))
    n_train = n - n_val - n_test
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [n_train, n_val, n_test], generator=generator)


def _get_element_refs(train_set, element_types, resume=None):
    """Fit training-only reference energies, or retain a resumed model's baseline."""
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        refs = checkpoint["state_dict"].get("model.element_refs.property_offset")
        if refs is None:
            print("Resuming a checkpoint without elemental references; preserving its original energy baseline.", flush=True)
            return None
        refs = refs.detach().cpu().clone()
        if refs.shape != (len(element_types),) or not torch.isfinite(refs).all():
            raise ValueError("Checkpoint elemental references must be finite and match the model's element count")
        print("Restoring elemental reference energies from the checkpoint.", flush=True)
        return refs

    dataset = train_set.dataset
    graphs = [dataset.graphs[i] for i in train_set.indices]
    energies = np.asarray([dataset.labels["energies"][i] for i in train_set.indices], dtype=np.float64)
    atom_ref = AtomRef(max_z=len(element_types))
    atom_ref.fit(graphs, energies)
    print(f"Fitted elemental reference energies from {len(train_set)} training structures.", flush=True)
    return atom_ref.property_offset.detach().clone()


def _load_dataset(args):
    if args.processed_cache:
        if args.max_structures is not None:
            raise ValueError("--processed-cache cannot be combined with --max-structures")
        cache = os.path.abspath(args.processed_cache)
        for filename in ("pyg_graph.pt", "lattice.pt", "state_attr.pt", "labels.json"):
            if not os.path.isfile(os.path.join(cache, filename)):
                raise FileNotFoundError(os.path.join(cache, filename))
        return DIEPDataset(
            raw_dir=os.path.dirname(cache), directory_name=os.path.basename(cache),
            include_line_graph=True, save_cache=False, mmap_cache=True,
        )
    structures, labels = load_datasets(
        data_dir=args.data_dir,
        datasets=args.datasets,
        matpes_functional=args.matpes_functional,
        matpes_version=args.matpes_version,
        max_structures=args.max_structures,
        force_limit=args.force_limit,
    )

    cache_suffix = ""
    if args.max_atoms is not None:
        keep = [i for i, s in enumerate(structures) if len(s) <= args.max_atoms]
        print(f"Filtering structures: keeping {len(keep)}/{len(structures)} with <= {args.max_atoms} atoms")
        structures = [structures[i] for i in keep]
        for k in labels:
            labels[k] = [labels[k][i] for i in keep]
        cache_suffix = f"_maxatoms{args.max_atoms}"

    # Keep the existing full MatPES cache, but isolate subsets and other conversion
    # settings so a smoke test cannot load or overwrite the full training dataset.
    cache_config = {
        k: getattr(args, k)
        for k in (
            "datasets", "matpes_functional", "matpes_version", "max_structures",
            "force_limit", "cutoff", "threebody_cutoff",
        )
    }
    if cache_config != dict(
        datasets="matpes", matpes_functional="PBE", matpes_version="2025.2",
        max_structures=None, force_limit=None, cutoff=5.0, threebody_cutoff=4.0,
    ):
        cache_suffix += "_" + hashlib.sha256(json.dumps(cache_config, sort_keys=True).encode()).hexdigest()[:12]

    element_types = DEFAULT_ELEMENTS
    converter = Structure2Graph(element_types=element_types, cutoff=args.cutoff)
    dataset = DIEPDataset(
        structures=structures,
        labels=labels,
        converter=converter,
        threebody_cutoff=args.threebody_cutoff,
        include_line_graph=True,
        # Only rank zero writes a newly processed cache.
        save_cache=int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0,
        raw_dir=os.path.join(args.data_dir, f"processed{cache_suffix}"),
    )

    return dataset


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger = CSVLogger(save_dir=args.save_dir, name="logs", version=args.log_version)
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1, monitor="val_Total_Loss", mode="min", filename="{epoch:04d}-best_model"
    )
    # A separate, unmonitored checkpoint advances even when validation does not
    # improve (this Lightning version otherwise leaves save_last at the best epoch).
    recovery_checkpoint = ModelCheckpoint(
        save_top_k=1,
        monitor=None,
        filename="latest",
        every_n_epochs=1,
        save_on_train_epoch_end=True,
        enable_version_counter=False,
    )
    step_checkpoint = ModelCheckpoint(
        save_top_k=1, monitor=None, filename="latest_step", every_n_train_steps=500,
        every_n_epochs=0, save_on_train_epoch_end=False, enable_version_counter=False,
    )
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val_Total_Loss", mode="min", patience=args.patience),
            checkpoint_callback,
            recovery_checkpoint,
            step_checkpoint,
            TrainingProgress(),
        ],
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        gradient_clip_val=2.0,
        inference_mode=False,
        num_sanity_val_steps=args.sanity_val_steps,
        use_distributed_sampler=args.max_atoms_per_batch is None,
    )

    dataset = _load_dataset(args)
    element_types = DEFAULT_ELEMENTS

    train_set, val_set, test_set = _split(dataset, args.val_frac, args.test_frac, args.seed)
    element_refs = _get_element_refs(train_set, element_types, resume=args.resume)
    # Rank zero may construct its loaders before Lightning launches other ranks.
    # Use the requested device count in that case; torchrun sets WORLD_SIZE upfront.
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", trainer.num_devices * trainer.num_nodes))
    if rank == 0 and element_refs is not None:
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(os.path.join(logger.log_dir, "element_refs.json"), "w") as f:
            json.dump(dict(zip(element_types, element_refs.tolist(), strict=True)), f, indent=2)
    train_loader, val_loader, test_loader = MGLDataLoader(
        train_data=train_set,
        val_data=val_set,
        test_data=test_set,
        batch_size=args.batch_size,
        max_atoms_per_batch=args.max_atoms_per_batch,
        rank=rank,
        num_replicas=world_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    print(
        f"Rank {rank}/{world_size}: {len(train_loader)} train, {len(val_loader)} validation, "
        f"{len(test_loader)} test batches", flush=True,
    )

    model = DIEP(
        element_types=element_types,
        is_intensive=False,
        cutoff=args.cutoff,
        threebody_cutoff=args.threebody_cutoff,
        integral_mode=args.integral_mode,
        dim_node_embedding=args.dim_node_embedding,
        dim_edge_embedding=args.dim_edge_embedding,
        nblocks=args.nblocks,
        units=args.units,
    )
    xavier_init(model)

    forces = torch.cat([dataset[i][3]["forces"] for i in train_set.indices])
    rms_forces = torch.sqrt(torch.mean(torch.sum(forces**2, dim=1)))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1.0e-5, amsgrad=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 1.0e-2)

    lit_model = PotentialLightningModule(
        model=model,
        element_refs=element_refs,
        data_std=rms_forces,
        optimizer=optimizer,
        scheduler=scheduler,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        stress_weight=args.stress_weight,
        loss=args.loss,
        lr=args.lr,
        include_line_graph=True,
        sync_dist=world_size > 1,
    )
    trainer.fit(
        model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=args.resume
    )
    trainer.test(model=lit_model, dataloaders=test_loader)

    if trainer.is_global_zero:
        model_export_path = os.path.join(args.save_dir, "trained_model")
        os.makedirs(model_export_path, exist_ok=True)
        lit_model.model.save(model_export_path)
        print(f"Saved trained model to {model_export_path}")
    if torch.distributed.is_initialized():
        # Wait for rank zero's export and pending collectives before any worker exits.
        trainer.strategy.barrier("training_complete")
        torch.distributed.destroy_process_group()


def _parse_args():
    parser = argparse.ArgumentParser(description="Train DIEP (PyG) on MP PES data")
    parser.add_argument("--data-dir", default="data/mp_pes")
    parser.add_argument("--processed-cache", default=None,
                        help="Load an existing DIEPDataset directory directly; its filtering and cutoffs must match the run")
    parser.add_argument("--sanity-val-steps", type=int, default=2,
                        help="Validation batches to check before training; -1 checks the full validation split")
    parser.add_argument("--datasets", choices=["mpf", "matpes", "both"], default="both")
    parser.add_argument("--matpes-functional", default="PBE")
    parser.add_argument("--matpes-version", default="2025.2")
    parser.add_argument("--max-structures", type=int, default=None, help="per-dataset cap, for smoke runs")
    parser.add_argument("--force-limit", type=float, default=None, help="MPF-only max-force outlier filter")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--threebody-cutoff", type=float, default=4.0)
    parser.add_argument("--integral-mode", choices=["sum", "grid"], default="grid")
    parser.add_argument("--dim-node-embedding", type=int, default=64)
    parser.add_argument("--dim-edge-embedding", type=int, default=64)
    parser.add_argument("--nblocks", type=int, default=3)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument(
        "--max-atoms", type=int, default=None, help="drop structures with more atoms than this before training"
    )
    parser.add_argument(
        "--max-atoms-per-batch",
        type=int,
        default=None,
        help="cap each batch's total atom count instead of using a fixed --batch-size structure count",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--loss",
        choices=["mse_loss", "huber_loss", "smooth_l1_loss", "l1_loss"],
        default="mse_loss",
        help="huber_loss/smooth_l1_loss are far less sensitive to extreme-stress outlier structures than mse_loss",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--test-frac", type=float, default=0.05)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=1.0)
    parser.add_argument("--stress-weight", type=float, default=0.1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=lambda value: int(value) if value.isdigit() else value, default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default="logs/mp_pes_pyg")
    parser.add_argument("--log-version", default=None, help="Shared logger version for externally launched DDP ranks")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint (.ckpt) to resume training from")
    return parser.parse_args()


if __name__ == "__main__":
    main(_parse_args())
