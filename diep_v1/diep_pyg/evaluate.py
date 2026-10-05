"""Evaluate an exported model on one split of a fold.

    python -m diep_pyg.evaluate --root data                       # released model, fold-1 test split
    python -m diep_pyg.evaluate --root data --limit 500 --device cpu
    python -m diep_pyg.evaluate --root data --predictions test_preds.npz

By default the splits come from the model directory itself (``models/diep_fold1/fold1``),
so the only thing needed besides the model is the graph cache written by
``python -m diep_pyg.matpes build``.

Two kinds of RMSE are reported, because they differ:

* ``*_RMSE_logged`` reproduces what training logged: the mean of per-batch RMSEs, weighted
  as Lightning weights them. It depends on batch size and order, and it is the number in
  ``metrics.json``.
* ``*_RMSE`` is the RMSE over the whole split. Use this one for anything new.

MAEs need no such split: the logged per-batch means, weighted by element count, are the
global MAE. Energies are per atom (eV/atom), forces per Cartesian component (eV/A), and
stresses per tensor component (GPa).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from diep_pyg.graph.data import collate_fn_pes

from diep_pyg.matpes import DIEPConfig, default_root, open_graph_cache
from diep_pyg.pretrained import DEFAULT_MODEL_DIR, load_potential, model_info

__all__ = ["evaluate"]


def evaluate(model_dir: Path, cfg: DIEPConfig, artifacts: Path, *, split: str = "test",
             batch_size: int = 32, device: str = "cpu", limit: int | None = None,
             num_workers: int = 0, predictions: Path | None = None) -> dict:
    splits = json.loads((artifacts / "splits.json").read_text())
    meta = json.loads((artifacts / "meta.json").read_text())
    refs = np.load(artifacts / "element_refs.npy")

    pot = load_potential(model_dir, device=device, stress=True)
    got = pot.element_refs.property_offset.detach().cpu().numpy()
    if not np.allclose(got, refs, rtol=0, atol=1e-5):
        raise SystemExit(f"{model_dir}'s element refs differ from {artifacts}'s")
    if float(pot.model.cutoff) != float(meta["cutoff"]):
        raise SystemExit(f"model cutoff {pot.model.cutoff} != cache cutoff {meta['cutoff']}")

    dataset = open_graph_cache(cfg, meta)
    if len(dataset) != int(meta["n_structures"]):
        raise SystemExit(f"cache has {len(dataset)} graphs but meta.json says "
                         f"{meta['n_structures']}; the splits index the cache by position")
    indices = splits[split][:limit] if limit else splits[split]
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers,
                        collate_fn=partial(collate_fn_pes, include_line_graph=True, include_stress=True))

    rows: dict[str, list] = {k: [] for k in ("n_atoms", "e_true", "e_pred", "f_mae", "f_max_err",
                                               "f_max_pred", "s_true", "s_pred")}
    # Running sums for the logged-style (batch-mean) RMSEs, weighted as Lightning weights them.
    logged = {"Energy": [0.0, 0], "Force": [0.0, 0], "Stress": [0.0, 0]}
    f_abs_sum = f_sq_sum = 0.0
    f_count = 0
    t0 = time.time()
    for b, (g, lat, state_attr, e_true, f_true, s_true) in enumerate(loader):
        g, lat, state_attr = g.to(device), lat.to(device), state_attr.to(device)
        with torch.enable_grad():
            e, f, s, _ = pot(g, lat, state_attr)
        e, f, s = e.detach().cpu(), f.detach().cpu(), s.detach().cpu()
        n_atoms = torch.bincount(g.batch.cpu(), minlength=len(e_true)).to(e.dtype)

        de = e_true / n_atoms - e / n_atoms
        df = f_true - f
        ds = s_true - s
        for key, d in (("Energy", de), ("Force", df), ("Stress", ds)):
            logged[key][0] += float(torch.sqrt((d**2).mean())) * d.numel()
            logged[key][1] += d.numel()
        f_abs_sum += float(df.abs().sum())
        f_sq_sum += float((df**2).sum())
        f_count += df.numel()

        bounds = np.concatenate([[0], np.cumsum(n_atoms.numpy().astype(int))])
        df_norm = df.norm(dim=1)
        f_norm = f.norm(dim=1)
        for i in range(len(e_true)):
            lo, hi = bounds[i], bounds[i + 1]
            rows["f_mae"].append(float(df[lo:hi].abs().mean()))
            rows["f_max_err"].append(float(df_norm[lo:hi].max()))
            rows["f_max_pred"].append(float(f_norm[lo:hi].max()))
        rows["n_atoms"].extend(n_atoms.int().tolist())
        rows["e_true"].extend((e_true / n_atoms).tolist())
        rows["e_pred"].extend((e / n_atoms).tolist())
        rows["s_true"].extend(s_true.view(-1, 3, 3).numpy())
        rows["s_pred"].extend(s.view(-1, 3, 3).numpy())
        if b % 50 == 0:
            print(f"  batch {b + 1}/{len(loader)}  {time.time() - t0:.0f} s", flush=True)

    e_true_a, e_pred_a = np.array(rows["e_true"]), np.array(rows["e_pred"])
    s_true_a, s_pred_a = np.array(rows["s_true"]), np.array(rows["s_pred"])
    de_a, ds_a = e_true_a - e_pred_a, s_true_a - s_pred_a
    result = {
        "model": str(model_dir),
        "split": split,
        "n_structures": len(indices),
        "n_atoms": int(np.sum(rows["n_atoms"])),
        "Energy_MAE": float(np.abs(de_a).mean()),
        "Energy_RMSE": float(np.sqrt((de_a**2).mean())),
        "Energy_RMSE_logged": logged["Energy"][0] / logged["Energy"][1],
        "Force_MAE": f_abs_sum / f_count,
        "Force_RMSE": float(np.sqrt(f_sq_sum / f_count)),
        "Force_RMSE_logged": logged["Force"][0] / logged["Force"][1],
        "Stress_MAE": float(np.abs(ds_a).mean()),
        "Stress_RMSE": float(np.sqrt((ds_a**2).mean())),
        "Stress_RMSE_logged": logged["Stress"][0] / logged["Stress"][1],
        "max_predicted_force": float(np.max(rows["f_max_pred"])),
        "batch_size": batch_size,
        "seconds": round(time.time() - t0, 1),
    }
    if predictions:
        np.savez_compressed(predictions, index=np.asarray(indices),
                            **{k: np.asarray(v) for k, v in rows.items()})
        result["predictions"] = str(predictions)
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--root", type=Path, default=None,
                   help="data root holding the graph cache; defaults to $DIEP_MATPES_ROOT, else ./data")
    p.add_argument("--cache-name", default=DIEPConfig.cache_name)
    p.add_argument("--artifacts", type=Path, default=None,
                   help="fold dir with splits.json/meta.json/element_refs.npy; defaults to "
                        "the model's own fold{k} directory")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=32,
                   help="32 matches training, which matters only for the *_RMSE_logged values")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="first N structures of the split")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--predictions", type=Path, default=None,
                   help="write per-structure predictions to this .npz")
    p.add_argument("--out", type=Path, default=None, help="write the metrics to this .json")
    args = p.parse_args(argv)

    artifacts = args.artifacts
    if artifacts is None:
        fold = model_info(args.model).get("fold")
        if fold is None:
            raise SystemExit("--artifacts is required: the model has no PROVENANCE.json fold")
        artifacts = args.model / f"fold{fold}"
    cfg = DIEPConfig(root=args.root or default_root(), cache_name=args.cache_name)
    result = evaluate(args.model, cfg, artifacts, split=args.split, batch_size=args.batch_size,
                      device=args.device, limit=args.limit, num_workers=args.num_workers,
                      predictions=args.predictions)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
