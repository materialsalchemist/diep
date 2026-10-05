"""Export a training checkpoint to a self-contained model directory.

    python -m diep_pyg.export \\
        --ckpt data/runs_diep/diep_fold1/checkpoints/best-epoch=0295.ckpt \\
        --artifacts data/artifacts_full/fold1 \\
        --metrics data/runs_diep/diep_fold1/metrics.json \\
        --out models/diep_fold1

This is how ``models/diep_fold1`` was written. The output (``model.json``, ``model.pt``,
``state.pt``, ``PROVENANCE.json``) loads with ``Potential.load`` and needs neither Lightning
nor the training data.

The state dict is loaded **strictly and by hand**, not through Lightning's
``load_from_checkpoint``. ``PotentialLightningModule.on_load_checkpoint`` backfills any key
the checkpoint lacks, so a canonical-frame checkpoint loaded into a bond-frame model would
silently gain the ``bond_anchored_triplets`` marker and run with the wrong descriptors.

The export is verified before it is written and again after it is read back: every tensor
must match the checkpoint exactly, and energies, forces and stresses on test structures
must be bit-identical to the checkpoint's own module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

__all__ = ["load_checkpoint_potential", "export"]

#: Architecture of the released model: ``train``'s defaults, which ``slurm/train.sh`` does
#: not override. A shape mismatch fails the strict load. The grid geometry has no
#: parameters, so the parameter-count check against metrics.json is what ties it to the run.
ARCH = dict(nblocks=3, units=64, dim_node_embedding=64, dim_edge_embedding=64,
            grid_half_length=5.0, base_spacing=1.0, gaussian_sigma=1.0,
            softening_epsilon=0.5, no_effective_charge=False)


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_potential(ckpt: Path, artifacts: Path, run: dict):
    """Rebuild the training module from ``ckpt`` and return ``(potential, blob)``.

    ``run`` supplies ``use_triplets``, ``integral_mode``, ``triplet_frame`` and
    ``n_params`` (from metrics.json, or the training defaults).
    """
    from diep_pyg.utils.training import PotentialLightningModule

    from diep_pyg.train import build_model

    meta = json.loads((artifacts / "meta.json").read_text())
    refs = np.load(artifacts / "element_refs.npy")
    args = SimpleNamespace(**ARCH, no_triplets=not run["use_triplets"],
                           integral_mode=run["integral_mode"], triplet_frame=run["triplet_frame"])
    model = build_model(meta, args)
    n_params = sum(p.numel() for p in model.parameters())
    if run.get("n_params") is not None and n_params != run["n_params"]:
        raise SystemExit(f"rebuilt model has {n_params} params, the run had {run['n_params']}")

    # stress_weight must be non-zero: the module derives Potential.calc_stresses from it,
    # and with stress off the potential returns a zeros(1) placeholder instead of a stress.
    weights = run.get("weights", {})
    module = PotentialLightningModule(
        model=model, element_refs=refs, data_std=float(meta["force_rms"]),
        include_line_graph=bool(run["use_triplets"]),
        energy_weight=float(weights.get("energy", 1.0)),
        force_weight=float(weights.get("force", 1.0)),
        stress_weight=float(weights.get("stress", 0.1)) or 0.1,
    )
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    module.load_state_dict(blob["state_dict"], strict=True)

    pot = module.model
    got = pot.element_refs.property_offset.detach().cpu().numpy()
    if not np.allclose(got, refs, rtol=0, atol=1e-5):
        raise SystemExit("checkpoint element_refs differ from the artifacts'; wrong fold?")
    if abs(float(pot.data_std) - float(meta["force_rms"])) > 1e-5:
        raise SystemExit(f"checkpoint data_std {float(pot.data_std)} != fold force_rms "
                         f"{meta['force_rms']}; wrong fold?")
    return pot, blob, meta, refs


def _probe_structures():
    """Small cells covering one element, several elements, and a skewed cell."""
    from pymatgen.core import Lattice, Structure

    # Seeded here rather than through Structure.perturb, which is unseeded by default.
    rng = np.random.default_rng(0)
    si_frac = np.array([
        [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
        [0.25, 0.25, 0.25], [0.75, 0.75, 0.25], [0.75, 0.25, 0.75], [0.25, 0.75, 0.75]])
    si = Structure(Lattice.cubic(5.43), ["Si"] * 8, si_frac + rng.normal(0, 0.05 / 5.43, (8, 3)))
    nacl = Structure(Lattice.cubic(5.64), ["Na", "Cl"] * 4, [
        [0, 0, 0], [0.5, 0, 0], [0.5, 0.5, 0], [0, 0.5, 0],
        [0.5, 0, 0.5], [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]])
    tri = Structure(Lattice.from_parameters(4.1, 4.6, 5.2, 81, 97, 112),
                    ["Li", "Fe", "P", "O", "O", "O"], rng.random((6, 3)))
    return [si, nacl, tri]


def _predict(pot, structure):
    from diep_pyg.graph.converters import Structure2Graph

    graph, lattice, state_attr = Structure2Graph(pot.model.element_types, pot.model.cutoff).get_graph(structure)
    e, f, s, _ = pot(graph, lattice, torch.tensor(state_attr))
    return e.detach(), f.detach(), s.detach()


def _assert_same_outputs(a, b, structures, label: str) -> None:
    # One thread: multithreaded CPU backward accumulates in a nondeterministic order, so
    # the *same* model's forces differ between two calls (measured up to 1e-4 eV/A on a
    # random cell with 4 threads, exactly 0 with 1). Bitwise comparison needs determinism.
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for i, structure in enumerate(structures):
            for name, x, y in zip(("energy", "forces", "stress"),
                                  _predict(a, structure), _predict(b, structure)):
                if not torch.equal(x, y):
                    raise SystemExit(f"{label}: {name} differs on probe {i} "
                                     f"(max |diff| {float((x - y).abs().max()):.3e})")
    finally:
        torch.set_num_threads(threads)


def _assert_same_state(a, b, label: str) -> int:
    sa, sb = a.state_dict(), b.state_dict()
    if sa.keys() != sb.keys():
        raise SystemExit(f"{label}: state-dict keys differ: {sorted(set(sa) ^ set(sb))}")
    for k in sa:
        if not torch.equal(sa[k].cpu(), sb[k].cpu()):
            raise SystemExit(f"{label}: tensor {k} differs")
    return len(sa)


def export(ckpt: Path, artifacts: Path, out: Path, metrics: Path | None, name: str) -> dict:
    from diep_pyg.apps.pes import Potential

    run = json.loads(metrics.read_text()) if metrics else {}
    run.setdefault("use_triplets", True)
    run.setdefault("integral_mode", "grid")
    # metrics.json written before the bond-anchored frame existed has no key; those runs
    # were all canonical.
    run.setdefault("triplet_frame", "canonical")

    src, blob, meta, refs = load_checkpoint_potential(ckpt, artifacts, run)
    src.eval()

    # A fresh Potential with plain-tensor init args, so model.pt holds no numpy objects and
    # loads under torch.load(weights_only=True) as well as the default.
    pot = Potential(
        model=src.model,
        element_refs=torch.tensor(refs, dtype=torch.float32),
        data_mean=0.0,
        data_std=float(meta["force_rms"]),
        calc_stresses=True,
        use_edges=bool(run["use_triplets"]),
    )
    pot.load_state_dict(src.state_dict(), strict=True)
    pot.eval()
    probes = _probe_structures()
    n_tensors = _assert_same_state(src, pot, "export")
    _assert_same_outputs(src, pot, probes, "export")

    n_params = sum(p.numel() for p in pot.model.parameters())
    test = run.get("test", {})
    provenance = {
        "name": name,
        "description": "DIEP (PyG backend) interatomic potential trained on MatPES r2SCAN, "
                       f"fold {meta.get('fold')} (seed {meta.get('seed')}), 90/5/5 split",
        "source_checkpoint": ckpt.name,
        "source_checkpoint_md5": _md5(ckpt),
        "exported": date.today().isoformat(),
        "exported_by": "diep_pyg.export",
        "epoch": int(blob["epoch"]),
        "global_step": int(blob["global_step"]),
        "fold": meta.get("fold"),
        "functional": meta.get("functional"),
        "architecture": {
            "model": "DIEP",
            "n_element_types": len(meta["element_types"]),
            "cutoff": float(meta["cutoff"]),
            "threebody_cutoff": float(meta["threebody_cutoff"]),
            **{k: v for k, v in ARCH.items() if k != "no_effective_charge"},
            "use_effective_charge": not ARCH["no_effective_charge"],
            "use_triplets": bool(run["use_triplets"]),
            "integral_mode": run["integral_mode"],
            "triplet_frame": run["triplet_frame"],
        },
        "n_parameters": int(n_params),
        "n_tensors_verified": n_tensors,
        "data_std_force_rms": float(meta["force_rms"]),
        "element_refs": "isolated-atom r2SCAN energies (MatPES-R2SCAN-atoms), eV",
        "units": {"energy": "eV", "forces": "eV/A", "stress": "GPa from Potential, "
                  "eV/A^3 from diep_pyg.pretrained.make_calculator"},
        "training": {k: run[k] for k in ("seed", "max_epochs", "scheduler_t_max", "batch_size",
                                         "loss", "weights", "n_train", "n_val", "n_test",
                                         "best_val_loss", "epochs_run", "early_stopped")
                     if k in run},
        "test_metrics": test,
    }
    pot.save(out, metadata={"name": name, "provenance": "PROVENANCE.json"})
    (out / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2))

    # Read it back through the public loader and check again, so what is on disk is what
    # was verified.
    reloaded = Potential.load(out)
    reloaded.eval()
    _assert_same_state(src, reloaded, "reload")
    _assert_same_outputs(src, reloaded, probes, "reload")
    torch.load(out / "model.pt", map_location="cpu", weights_only=True)
    print(f"exported {name}: epoch {provenance['epoch']}, {n_params:,} params, "
          f"{n_tensors} tensors verified, outputs bit-identical -> {out}")
    return provenance


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True, help="Lightning checkpoint (best-epoch=*.ckpt)")
    p.add_argument("--artifacts", type=Path, required=True,
                   help="fold artifacts dir holding meta.json and element_refs.npy")
    p.add_argument("--metrics", type=Path, default=None,
                   help="the run's metrics.json (architecture flags and test metrics)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default=None, help="model name; defaults to the --out basename")
    args = p.parse_args(argv)
    export(args.ckpt, args.artifacts, args.out, args.metrics, args.name or args.out.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
