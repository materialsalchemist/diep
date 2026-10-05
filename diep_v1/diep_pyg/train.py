"""Train one DIEP PES cell on a MatPES fold.

    python -m diep_pyg.train --fold 1
    python -m diep_pyg.train --fold 0 --limit-train 2000 --max-epochs 2
    python -m diep_pyg.train --fold 0 --no-triplets --tag notrip

This is the script that trained the released ``diep_fold1`` model; the exact
command line is in ``slurm/train.sh``. Two things worth knowing:

* **Three-body index space.** The PyG backend builds the three-body line graph in
  parent-bond index space by construction, so it does not have the M3GNet/matgl
  three-body index defect. ``--assert-invariants`` re-checks that on real batches.
* **The scheduler steps once per epoch.** ``PotentialLightningModule`` returns
  ``[opt], [sched]``, which Lightning steps per epoch, so ``--scheduler-t-max``
  is in epochs and the cosine actually anneals to ``1e-2 * lr``.

Resumes from ``checkpoints/last.ckpt`` automatically; pass ``--no-resume`` to
start fresh. ``last.ckpt`` is only rewritten when ``val_Total_Loss`` improves, so a
resume restarts from the best epoch so far, not from the last epoch run.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import CSVLogger
from torch.optim.lr_scheduler import CosineAnnealingLR

from diep_pyg.graph.compute import assert_lg_invariants
from diep_pyg.models import DIEP
from diep_pyg.utils.training import PotentialLightningModule, xavier_init

from diep_pyg.matpes import DIEPConfig, default_root, load_artifacts, make_loaders

logger = logging.getLogger("diep_pyg.train")


class InvariantCheckCallback(Callback):
    """Assert the line graph stays in parent-bond space, on real training batches.

    The check is cheap but not free, so it runs on the first ``n_batches`` of each
    epoch rather than all of them. It exists because the whole premise of this arm
    is that the PyG backend does not have the index-space defect; if that ever
    stopped being true, the run should fail loudly rather than quietly train a
    model whose three-body channel is wired to the wrong bonds.
    """

    def __init__(self, n_batches: int = 3) -> None:
        self.n_batches = n_batches

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if batch_idx >= self.n_batches:
            return
        g = batch[0]
        if getattr(g, "triple_index", None) is None:
            # The shared MatPES cache is written with include_line_graph=False, so the
            # collated batch carries no triples and this check used to return here --
            # silently verifying nothing for the entire run, which is the opposite of what
            # --assert-invariants promises. The model builds the line graph on the batched
            # graph inside forward(), so by on_train_batch_end it is attached to `g` and
            # this branch is only reached when the three-body channel is genuinely off.
            if getattr(pl_module.model.model, "use_edges", False):
                raise RuntimeError(
                    "three-body channel is enabled but the batch carries no triple_index "
                    "after the forward pass; the line graph was never built"
                )
            return
        assert_lg_invariants(g)
        # No triple may span two structures in the batch: triple_index holds bond
        # ids, and PyG offsets them via DIEPData.__inc__ during collation.
        src = g.edge_index[0]
        batch_of_bond = g.batch[src]
        ti = g.triple_index
        if not bool((batch_of_bond[ti[0]] == batch_of_bond[ti[1]]).all()):
            raise RuntimeError("cross-structure triple in a batched line graph")


def cell_name(fold: int, tag: str = "") -> str:
    name = f"diep_fold{fold}"
    return f"{name}_{tag}" if tag else name


def build_model(meta: dict, args) -> DIEP:
    """An extensive DIEP suitable for wrapping in ``Potential``.

    ``is_intensive=False`` and no set2set readout: the PES head sums per-atom
    energies. Cutoffs come from the build metadata, never from CLI defaults, so
    the model matches the graphs in the cache.
    """
    return DIEP(
        element_types=tuple(meta["element_types"]),
        is_intensive=False,
        cutoff=float(meta["cutoff"]),
        threebody_cutoff=float(meta["threebody_cutoff"]),
        nblocks=args.nblocks,
        units=args.units,
        dim_node_embedding=args.dim_node_embedding,
        dim_edge_embedding=args.dim_edge_embedding,
        # use_triplets and use_edges are collapsed onto one flag inside DIEP; pass
        # the one that is documented as the switch.
        use_triplets=not args.no_triplets,
        integral_mode=args.integral_mode,
        grid_half_length=args.grid_half_length,
        base_spacing=args.base_spacing,
        gaussian_sigma=args.gaussian_sigma,
        softening_epsilon=args.softening_epsilon,
        use_effective_charge=not args.no_effective_charge,
        # getattr: callers that build their own Namespace (evaluation scripts) predate the
        # flag, and every checkpoint they load was trained canonical.
        triplet_frame=getattr(args, "triplet_frame", "canonical"),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=None,
                   help="data root; defaults to $DIEP_MATPES_ROOT, else ./data")
    p.add_argument("--cache-name", default=DIEPConfig.cache_name,
                   help="graph cache directory under --root (default: %(default)s)")
    p.add_argument("--fold", type=int, default=0, help="which artifacts_full/fold{k} to train on")
    p.add_argument("--artifacts-name", default=None,
                   help="override the artifacts dir; defaults to artifacts_full/fold{--fold}")
    p.add_argument("--tag", default="", help="suffix for the run directory")
    p.add_argument("--out-dir", type=Path, default=None)

    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--loss", default="l1_loss",
                   choices=["l1_loss", "mse_loss", "huber_loss", "smooth_l1_loss"])
    p.add_argument("--energy-weight", type=float, default=1.0)
    p.add_argument("--force-weight", type=float, default=1.0)
    p.add_argument("--stress-weight", type=float, default=0.1)
    p.add_argument("--accumulate-grad-batches", type=int, default=4)
    p.add_argument("--gradient-clip-val", type=float, default=2.0)
    # Lightning steps this scheduler once per epoch (verified), so T_max is in
    # epochs and the cosine reaches eta_min at --max-epochs. This is NOT the
    # M3GNet situation, where the scheduler stepped twice per epoch.
    p.add_argument("--scheduler-t-max", type=int, default=None,
                   help="CosineAnnealingLR T_max in epochs; defaults to --max-epochs")
    p.add_argument("--init-from", type=Path, default=None,
                   help="load model weights from this checkpoint, then start a "
                        "fresh optimizer/scheduler/epoch counter")

    # Architecture.
    p.add_argument("--nblocks", type=int, default=3)
    p.add_argument("--units", type=int, default=64)
    p.add_argument("--dim-node-embedding", type=int, default=64)
    p.add_argument("--dim-edge-embedding", type=int, default=64)
    p.add_argument("--no-triplets", action="store_true",
                   help="disable the three-body channel entirely (ablation)")
    # DIEP integrator geometry.
    p.add_argument("--integral-mode", default="grid", choices=["grid", "sum"])
    p.add_argument("--grid-half-length", type=float, default=5.0)
    p.add_argument("--base-spacing", type=float, default=1.0)
    p.add_argument("--gaussian-sigma", type=float, default=1.0)
    p.add_argument("--softening-epsilon", type=float, default=0.5)
    p.add_argument("--no-effective-charge", action="store_true",
                   help="use Z instead of sqrt(Z)")
    p.add_argument("--triplet-frame", default="canonical", choices=["canonical", "bond"],
                   help="'bond' draws each triplet in its receiving bond's frame (smooth "
                        "energy, no ordering steps). Must match the checkpoint when resuming.")

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-mmap", action="store_true", help="copy the cache instead of mmapping it")
    p.add_argument("--accelerator", default="auto")
    p.add_argument("--devices", default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--progress-bar", action="store_true", default=True)
    p.add_argument("--no-progress-bar", dest="progress_bar", action="store_false",
                   help="quieter logs, for batch jobs")
    p.add_argument("--limit-train", type=int, default=None, help="cap training rows (smoke test)")
    p.add_argument("--limit-eval", type=int, default=None, help="cap val/test rows (smoke test)")
    p.add_argument("--assert-invariants", action="store_true", default=True,
                   help="check the line graph's index space on the first batches of each epoch")
    p.add_argument("--no-assert-invariants", dest="assert_invariants", action="store_false")
    p.add_argument("--fast-dev-run", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    pl.seed_everything(args.seed, workers=True)

    artifacts_name = args.artifacts_name or f"artifacts_full/fold{args.fold}"
    cfg = DIEPConfig(root=args.root or default_root(), artifacts_name=artifacts_name,
                     cache_name=args.cache_name)
    splits, meta, element_refs = load_artifacts(cfg)
    logger.info("artifacts %s | train/val/test = %d/%d/%d | force RMS %.4f",
                cfg.artifacts_dir, len(splits["train"]), len(splits["val"]),
                len(splits["test"]), float(meta["force_rms"]))
    logger.info("cutoff %.2f, threebody_cutoff %.2f, %d element types",
                float(meta["cutoff"]), float(meta["threebody_cutoff"]), len(meta["element_types"]))

    name = cell_name(args.fold, args.tag)
    out_dir = args.out_dir or (cfg.runs_dir / name)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("cell %s -> %s", name, out_dir)
    if args.no_triplets:
        logger.warning("three-body channel DISABLED (--no-triplets): this is an ablation, "
                       "not the paired DIEP baseline")

    loaders = make_loaders(
        cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_stress=args.stress_weight != 0,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        mmap=not args.no_mmap,
    )

    model = build_model(meta, args)
    xavier_init(model)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("DIEP: %s params, use_triplets=%s, integral_mode=%s, triplet_frame=%s",
                f"{n_params:,}", model.use_triplets, args.integral_mode, args.triplet_frame)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True
    )
    t_max = args.scheduler_t_max or args.max_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=1e-2 * args.lr)
    logger.info("cosine T_max = %d epochs (stepped once per epoch; lr %.2e -> %.2e)",
                t_max, args.lr, 1e-2 * args.lr)

    module = PotentialLightningModule(
        model=model,
        element_refs=np.asarray(element_refs),
        data_std=float(meta["force_rms"]),
        include_line_graph=not args.no_triplets,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        stress_weight=args.stress_weight,
        loss=args.loss,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    if args.init_from:
        # strict=True: this module is rebuilt from the same meta.json and the same
        # fold artifacts, so every key must match. A silent partial load here would
        # look like a bad schedule rather than a bad checkpoint.
        blob = torch.load(args.init_from, map_location="cpu", weights_only=False)
        module.load_state_dict(blob["state_dict"], strict=True)
        logger.info("init-from %s (source epoch %s) -- weights only; optimizer, "
                    "scheduler and epoch counter start fresh", args.init_from, blob.get("epoch"))

    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir), monitor="val_Total_Loss", mode="min", save_top_k=1,
        save_last=True, filename="best-{epoch:04d}",
    )
    callbacks: list[Callback] = [
        checkpoint,
        EarlyStopping(monitor="val_Total_Loss", mode="min", patience=args.patience),
    ]
    if args.assert_invariants:
        callbacks.append(InvariantCheckCallback())
    if args.progress_bar:
        callbacks.append(TQDMProgressBar(refresh_rate=20))

    trainer = pl.Trainer(
        logger=CSVLogger(save_dir=str(out_dir), name="logs"),
        callbacks=callbacks,
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches,
        enable_progress_bar=args.progress_bar,
        fast_dev_run=args.fast_dev_run,
        # MANDATORY for a PES model. Lightning's eval loops default to
        # torch.inference_mode(), whose tensors are permanently barred from
        # autograd -- unlike no_grad, it cannot be lifted by the
        # torch.set_grad_enabled(True) that the step relies on. Forces are dE/dx,
        # so Potential.forward calls grad() and dies with "element 0 of tensors
        # does not require grad". Training survives because its loop never enters
        # inference_mode; validation and test do.
        inference_mode=False,
    )

    resume = ckpt_dir / "last.ckpt"
    ckpt_path = str(resume) if resume.exists() and not args.no_resume else None
    if ckpt_path:
        logger.info("resuming from %s", ckpt_path)

    trainer.fit(module, loaders.train, loaders.val, ckpt_path=ckpt_path)

    if args.fast_dev_run:
        logger.info("fast-dev-run: skipping test evaluation")
        return 0

    logger.info("testing on the held-out split")
    (metrics,) = trainer.test(module, dataloaders=loaders.test, ckpt_path="best")

    summary = {
        "model": "DIEP",
        "backend": "pyg",
        "fold": args.fold,
        "artifacts": artifacts_name,
        "tag": args.tag,
        "seed": args.seed,
        "use_triplets": bool(model.use_triplets),
        "integral_mode": args.integral_mode,
        "triplet_frame": args.triplet_frame,
        "n_params": int(n_params),
        "max_epochs": args.max_epochs,
        "scheduler_t_max": t_max,
        "batch_size": args.batch_size,
        "loss": args.loss,
        "weights": {"energy": args.energy_weight, "force": args.force_weight,
                    "stress": args.stress_weight},
        "n_train": len(loaders.train.dataset),
        "n_val": len(loaders.val.dataset),
        "n_test": len(loaders.test.dataset),
        "best_val_loss": float(checkpoint.best_model_score) if checkpoint.best_model_score else None,
        "best_checkpoint": checkpoint.best_model_path,
        "epochs_run": int(trainer.current_epoch),
        "early_stopped": int(trainer.current_epoch) < args.max_epochs,
        "test": metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote %s", out_dir / "metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
