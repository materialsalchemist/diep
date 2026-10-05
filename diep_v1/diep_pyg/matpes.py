"""MatPES r2SCAN as DIEP training data: graph cache, fold splits and loaders.

    python -m diep_pyg.matpes build --root data --json MatPES-R2SCAN-2025.1.json.gz \\
                                    --atoms MatPES-R2SCAN-atoms.json.gz
    python -m diep_pyg.matpes build --root data --limit 2000 --cache-name smoke   # smoke test
    python -m diep_pyg.matpes folds --root data --folds 3

The same steps from Python are :func:`build_cache` then :func:`build_folds`; training reads
a fold back through :func:`make_loaders`. Everything lives under one data root::

    <root>/
        MatPES-R2SCAN-2025.1.json.gz     the source file (``build`` downloads it if not given)
        MatPES-R2SCAN-atoms.json.gz      isolated-atom energies (the element refs)
        DIEPDataset/                     converted-graph cache (``build_cache``)
        artifacts/element_refs.npy       written by ``build_cache``
        artifacts_full/fold{k}/          splits.json, meta.json, element_refs.npy (``build_folds``)
        runs_diep/diep_fold{k}/          one training cell (``diep_pyg.train``)

The root defaults to ``$DIEP_MATPES_ROOT``, else ``./data``. Nothing writes into the cache
or the fold artifacts once they exist; training only writes under ``runs_diep``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from diep_pyg.config import DEFAULT_ELEMENTS
from diep_pyg.graph.converters import Structure2Graph
from diep_pyg.graph.data import DIEPDataset, collate_fn_pes

__all__ = [
    "DIEPConfig",
    "DIEPLoaders",
    "build_cache",
    "build_folds",
    "default_root",
    "element_refs_from_atoms_file",
    "load_artifacts",
    "load_matpes",
    "make_loaders",
    "open_graph_cache",
]


# --- paths -----------------------------------------------------------------------------


def default_root() -> Path:
    """``$DIEP_MATPES_ROOT`` if set, else ``./data`` relative to the working directory."""
    return Path(os.environ.get("DIEP_MATPES_ROOT", "data"))


@dataclass
class DIEPConfig:
    """Paths for one DIEP training cell."""

    root: Path = field(default_factory=default_root)

    #: Converted-graph cache: ``pyg_graph.pt`` (a list of PyG ``Data`` objects),
    #: ``lattice.pt``, ``state_attr.pt`` and ``labels.json``, as written by
    #: ``DIEPDataset``. The released model was trained from a cache named
    #: ``MGLDataset`` written by matgl 4.0.2's converter; :func:`build_cache`
    #: writes the same graphs with diep_pyg's converter (see the README).
    cache_name: str = "DIEPDataset"

    #: Fold artifacts written by :func:`build_folds`: splits.json, meta.json,
    #: element_refs.npy.
    artifacts_name: str = "artifacts_full/fold0"

    #: Run directories live under this name.
    runs_name: str = "runs_diep"

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    @property
    def cache_dir(self) -> Path:
        return self.root / self.cache_name

    @property
    def artifacts_dir(self) -> Path:
        return self.root / self.artifacts_name

    @property
    def splits_json(self) -> Path:
        return self.artifacts_dir / "splits.json"

    @property
    def meta_json(self) -> Path:
        return self.artifacts_dir / "meta.json"

    @property
    def element_refs_npy(self) -> Path:
        return self.artifacts_dir / "element_refs.npy"

    @property
    def runs_dir(self) -> Path:
        return self.root / self.runs_name


# --- MatPES json -> graph cache --------------------------------------------------------


def _load_records(source_json: Path | None, functional: str, limit: int | None) -> list[dict]:
    """Raw MatPES records, from a local file or via the ``matpes`` package."""
    if source_json is not None:
        from monty.serialization import loadfn

        print(f"reading {source_json}")
        records = list(loadfn(source_json))
        return records[:limit] if limit else records

    try:
        from matpes.data import get_data
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            "The 'matpes' package is not installed and --json was not given.\n"
            "  pip install matpes\n"
            f"or download MatPES-{functional}-2025.1.json.gz yourself and pass --json/--atoms."
        ) from exc

    print(f"downloading MatPES-{functional} via the matpes package")
    records = list(get_data(functional, download_atoms=True))
    return records[:limit] if limit else records


def _find_atoms_file(root: Path, functional: str, atoms_file: Path | None) -> Path:
    """Locate the isolated-atom energies file, under ``root`` or the cwd.

    ``matpes.get_data`` writes to whichever directory it was called from, and the name and
    format vary by release (``.json.gz`` for r2SCAN, ``.jsonl`` for the PBE tutorial).
    """
    if atoms_file is not None:
        if not Path(atoms_file).exists():
            raise SystemExit(f"--atoms {atoms_file} does not exist")
        return Path(atoms_file)
    stem = f"MatPES-{functional}-atoms"
    candidates = [
        directory / f"{stem}{suffix}"
        for directory in (root, Path.cwd())
        for suffix in (".json.gz", ".jsonl", ".json", ".jsonl.gz")
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = "\n  ".join(str(c) for c in dict.fromkeys(candidates))
    raise SystemExit(f"isolated-atom energies not found. Tried:\n  {tried}\nPass --atoms <path>.")


def element_refs_from_atoms_file(path: Path, element_types: tuple[str, ...]) -> np.ndarray:
    """Isolated-atom energies aligned to ``element_types``.

    Elements absent from the file get 0.0, which leaves them unreferenced rather than
    corrupting them: ``Potential`` subtracts ``element_refs`` from the total energy.
    """
    from monty.io import zopen

    with zopen(str(path), "rt", encoding="utf-8") as handle:
        text = handle.read()
    # One JSON array (r2SCAN) or one object per line (PBE tutorial); decide by parsing.
    try:
        records = json.loads(text)
        if isinstance(records, dict):
            records = list(records.values())
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    energies = {d["elements"][0]: d["energy"] for d in records}
    print(f"isolated-atom energies: {len(energies)} elements from {path.name}")
    missing = [e for e in element_types if e not in energies]
    if missing:
        print(f"warning: {len(missing)} elements have no isolated-atom energy (set to 0.0): "
              f"{', '.join(missing[:12])}{' ...' if len(missing) > 12 else ''}")
    return np.array([energies.get(e, 0.0) for e in element_types], dtype=float)


def load_matpes(records: list[dict]) -> tuple[list, dict[str, list]]:
    """Return ``(structures, labels)`` from raw MatPES records, in record order."""
    from ase.stress import voigt_6_to_full_3x3_stress
    from pymatgen.core import Structure

    functionals = {r.get("functional") for r in records if r.get("functional")}
    if functionals:
        print(f"functional(s) in source: {sorted(functionals)}")
        if not any("scan" in str(f).lower() for f in functionals):
            print("WARNING: source does not look like r2SCAN. Energies will not be on the "
                  "same reference as the r2SCAN element refs.")

    structures: list = []
    labels: dict[str, list] = collections.defaultdict(list)
    for record in tqdm(records, desc="parsing"):
        struct = record["structure"]
        structures.append(struct if isinstance(struct, Structure) else Structure.from_dict(struct))
        labels["energies"].append(record["energy"])
        labels["forces"].append(record["forces"])
        # Sign flip + kbar -> GPa, exactly as the MatPES tutorial does it.
        labels["stresses"].append(
            (voigt_6_to_full_3x3_stress(np.array(record["stress"])) * -0.1).tolist()
        )
    return structures, dict(labels)


def build_cache(cfg: DIEPConfig, *, functional: str, cutoff: float, source_json: Path | None,
                atoms_file: Path | None, limit: int | None) -> int:
    """Convert MatPES to the graph cache and write the element refs. Returns the count.

    Run once; every fold reads the same cache. Writes
    ``<root>/<cache-name>/{pyg_graph.pt, lattice.pt, state_attr.pt, labels.json,
    fingerprint.json}`` and ``<root>/artifacts/element_refs.npy``.

    Graphs are in the source file's record order, and the fold splits index into the cache
    by position, so the cache must be built from the unmodified MatPES file.

    Label convention is the MatPES tutorial's: stresses are
    ``voigt_6_to_full_3x3_stress(stress) * -0.1``, a sign flip and kbar -> GPa. Getting
    this wrong silently trains against the wrong tensor, so it happens in exactly one place
    (:func:`load_matpes`).

    The released model was trained from a cache written by matgl 4.0.2's PyG converter.
    This writes the same graphs with diep_pyg's converter, which calls the same
    ``find_points_in_spheres`` and keeps its ordering. The README records the check that the
    two are identical.
    """
    if (cfg.cache_dir / "pyg_graph.pt").exists():
        raise SystemExit(f"{cfg.cache_dir} already holds a cache; remove it or pass --cache-name")

    element_types = tuple(DEFAULT_ELEMENTS)
    # Resolved before the expensive conversion, so a missing atoms file fails in seconds.
    element_refs = element_refs_from_atoms_file(
        _find_atoms_file(cfg.root, functional, atoms_file), element_types)

    structures, labels = load_matpes(_load_records(source_json, functional, limit))
    n = len(structures)
    print(f"{n} structures; converting to graphs (cutoff={cutoff}) -> {cfg.cache_dir}")

    dataset = DIEPDataset(
        structures=structures,
        labels=labels,
        converter=Structure2Graph(element_types=element_types, cutoff=cutoff),
        # The model builds the three-body line graph per batch at its own threebody_cutoff,
        # so caching it would only add size (and a cutoff to keep in sync).
        include_line_graph=False,
        directory_name=cfg.cache_name,
        raw_dir=str(cfg.root),
        clear_processed=True,
        save_cache=False,
    )
    # Match the cache the released model was trained from, field for field:
    # * the converter also stores each lattice on its graph. It duplicates lattice.pt and the
    #   forward pass reads the lattice argument instead, so it is dropped;
    # * that cache (written by matgl 4.0.2) held edge_index as int32 and node_type as int64,
    #   the reverse of diep_pyg's converter. The values are identical either way; the dtypes
    #   are matched so a rebuilt cache is the training cache, not merely equivalent to it.
    for graph in dataset.graphs:
        del graph.lattice
        graph.edge_index = graph.edge_index.to(torch.int32)
        graph.node_type = graph.node_type.to(torch.int64)
    dataset.save_cache = True
    dataset.save()
    (cfg.cache_dir / "fingerprint.json").write_text(json.dumps({
        "converter_class": "diep_pyg.graph.converters.Structure2Graph",
        "cutoff": cutoff,
        "element_types": list(element_types),
        "include_line_graph": False,
        "n_structures": n,
        "functional": functional,
    }, indent=2))

    refs_dir = cfg.root / "artifacts"
    refs_dir.mkdir(parents=True, exist_ok=True)
    np.save(refs_dir / "element_refs.npy", element_refs)
    print(f"wrote {cfg.cache_dir} and {refs_dir / 'element_refs.npy'}")
    return n


# --- fold splits -----------------------------------------------------------------------


def build_folds(cache_dir: Path, out_dir: Path, *, n_folds: int, seed0: int,
                element_refs_src: Path, cutoff: float, threebody_cutoff: float,
                train_frac: float = 0.90, val_frac: float = 0.05,
                functional: str = "R2SCAN", overwrite: bool = False) -> list[dict]:
    """Write ``n_folds`` seeded train/val/test splits over the whole graph cache.

    Fold k is an independently seeded 90/5/5 draw (seed ``seed0 + k``), not a partition, so
    folds overlap (~81% of training rows shared between any two) and give replicate scatter
    rather than independent estimates. The released model is fold 1 (seed 43).

    Writes ``<out_dir>/fold{k}/{splits.json, meta.json, element_refs.npy}``, the layout
    ``diep_pyg.train --artifacts-name`` expects. ``meta.json`` carries ``force_rms``, the RMS
    force norm over that fold's *training* rows only, which ``Potential`` uses as
    ``data_std``. Fitting it on all rows would leak test statistics into the model's output
    scaling.

    Given the same cache this reproduces the shipped ``models/diep_fold1/fold1`` files
    exactly; ``tests/test_release.py`` checks that on a synthetic cache, and the README
    records the check on the real one.
    """
    # Every finished run's metrics index these splits; silently redrawing them under a run
    # would detach its test numbers from the rows they were measured on.
    existing = [out_dir / f"fold{k}" for k in range(n_folds) if (out_dir / f"fold{k}" / "splits.json").exists()]
    if existing and not overwrite:
        raise SystemExit(f"{', '.join(map(str, existing))} already exist; pass --overwrite "
                         "to replace them, or --out-name to write elsewhere")

    fingerprint_path = cache_dir / "fingerprint.json"
    if fingerprint_path.exists():
        fingerprint = json.loads(fingerprint_path.read_text())
        if float(fingerprint["cutoff"]) != float(cutoff):
            raise SystemExit(
                f"cache was built at cutoff {fingerprint['cutoff']}, --cutoff says {cutoff}; "
                "a mismatch silently describes the two-body graph wrongly"
            )
    else:
        print(f"warning: {fingerprint_path} not found; trusting --cutoff {cutoff}")

    element_types = tuple(DEFAULT_ELEMENTS)
    if not element_refs_src.exists():
        raise SystemExit(f"{element_refs_src} not found -- run `python -m diep_pyg.matpes build` first")
    element_refs = np.load(element_refs_src)
    if len(element_refs) != len(element_types):
        raise SystemExit(f"element_refs has {len(element_refs)} entries against "
                         f"{len(element_types)} elements")

    # Forces come from the cache's own label file, so the rows are guaranteed to be in the
    # same order as the graphs the split indices address.
    print(f"reading forces from {cache_dir / 'labels.json'}")
    forces = json.loads((cache_dir / "labels.json").read_text())["forces"]
    n = len(forces)
    print(f"{n} structures in the cache")

    metas: list[dict] = []
    for fold in range(n_folds):
        seed = seed0 + fold
        perm = np.random.default_rng(seed).permutation(n)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        splits = {
            "train": perm[:n_train].tolist(),
            "val": perm[n_train : n_train + n_val].tolist(),
            "test": perm[n_train + n_val :].tolist(),
        }
        assert not (set(splits["train"]) & set(splits["test"])), "train/test overlap"
        assert not (set(splits["train"]) & set(splits["val"])), "train/val overlap"

        train_forces = np.concatenate([np.asarray(forces[i]) for i in splits["train"]])
        force_rms = float(np.sqrt(np.mean(np.sum(train_forces**2, axis=1))))
        del train_forces

        meta = {
            "functional": functional,
            "n_structures": n,
            "element_types": list(element_types),
            "cutoff": cutoff,
            "threebody_cutoff": threebody_cutoff,
            "force_rms": force_rms,
            "seed": seed,
            "fold": fold,
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "source": "build_folds: seeded 90/5/5 over the full cache",
        }

        fold_dir = out_dir / f"fold{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "splits.json").write_text(json.dumps(splits))
        (fold_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        shutil.copyfile(element_refs_src, fold_dir / "element_refs.npy")

        sizes = meta["split_sizes"]
        print(f"  fold {fold} (seed {seed}): train {sizes['train']} / val {sizes['val']} "
              f"/ test {sizes['test']}  force RMS {force_rms:.4f} eV/A -> {fold_dir}")
        metas.append(meta)
    return metas


# --- reading a fold back ---------------------------------------------------------------


def load_artifacts(cfg: DIEPConfig) -> tuple[dict, dict, np.ndarray]:
    """Return ``(splits, meta, element_refs)`` written by :func:`build_folds`."""
    if not cfg.splits_json.exists():
        raise SystemExit(
            f"{cfg.splits_json} not found -- build the fold artifacts with "
            f"`python -m diep_pyg.matpes folds` first"
        )
    splits = json.loads(cfg.splits_json.read_text())
    meta = json.loads(cfg.meta_json.read_text())
    element_refs = np.load(cfg.element_refs_npy)
    return splits, meta, element_refs


def open_graph_cache(cfg: DIEPConfig, meta: dict, mmap: bool = True) -> DIEPDataset:
    """Open the shared converted-graph cache for reading.

    Cutoff and element list come from the build metadata rather than CLI defaults:
    the cached graphs were built with those exact values, and passing anything else
    would silently describe the cache wrongly.

    ``save_cache=False`` is important -- training opens the cache read-only and must
    never rewrite it (several folds may share one cache).
    """
    cache_dir = cfg.cache_dir
    if not (cache_dir / "pyg_graph.pt").exists():
        raise SystemExit(f"{cache_dir}/pyg_graph.pt not found -- build it with "
                         "`python -m diep_pyg.matpes build` first")
    return DIEPDataset(
        converter=Structure2Graph(element_types=tuple(meta["element_types"]), cutoff=float(meta["cutoff"])),
        threebody_cutoff=float(meta["threebody_cutoff"]),
        # The cache is written with include_line_graph=False. The model builds the
        # line graph itself on the batched graph (verified equivalent to building it
        # per-graph then batching), so nothing is lost by not having it cached.
        include_line_graph=True,
        directory_name=cfg.cache_name,
        raw_dir=str(cfg.root),
        save_cache=False,
        mmap_cache=mmap,
    )


@dataclass
class DIEPLoaders:
    """Train/val/test loaders over one fold."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    dataset: DIEPDataset


def make_loaders(
    cfg: DIEPConfig,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    include_stress: bool = True,
    limit_train: int | None = None,
    limit_eval: int | None = None,
    mmap: bool = True,
) -> DIEPLoaders:
    """Build the three loaders for one fold.

    Args:
        cfg: paths for this cell.
        batch_size: structures per batch.
        num_workers: dataloader workers.
        include_stress: whether stress labels ride along in the collate.
        limit_train: cap training rows, for smoke tests.
        limit_eval: cap val/test rows, for smoke tests.
        mmap: memory-map the cached tensors rather than copying them.
    """
    splits, meta, _ = load_artifacts(cfg)
    dataset = open_graph_cache(cfg, meta, mmap=mmap)
    if len(dataset) != int(meta["n_structures"]):
        raise SystemExit(
            f"cache has {len(dataset)} graphs but meta.json says {meta['n_structures']} -- "
            "the splits index into the cache by position, so these must agree"
        )

    collate = partial(collate_fn_pes, include_line_graph=True, include_stress=include_stress)

    def build(indices: list[int], shuffle: bool, limit: int | None) -> DataLoader:
        if limit is not None:
            indices = indices[:limit]
        return DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate,
            # Workers each mmap the cache; persisting them avoids re-opening it
            # every epoch, which on a 2.5 GB cache is the difference between a
            # few seconds and a few minutes of per-epoch overhead.
            persistent_workers=num_workers > 0,
            pin_memory=True,
            drop_last=False,
        )

    return DIEPLoaders(
        train=build(splits["train"], True, limit_train),
        val=build(splits["val"], False, limit_eval),
        test=build(splits["test"], False, limit_eval),
        dataset=dataset,
    )


# --- command line ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="MatPES json -> graph cache + element refs",
                                description=build_cache.__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    folds = commands.add_parser("folds", help="seeded 90/5/5 splits, per-fold meta.json",
                                description=build_folds.__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    for q in (build, folds):
        q.add_argument("--root", type=Path, default=None,
                       help="data root; defaults to $DIEP_MATPES_ROOT, else ./data")
        q.add_argument("--cache-name", default=DIEPConfig.cache_name)

    build.add_argument("--functional", default="R2SCAN", help="R2SCAN (the released model) or PBE")
    build.add_argument("--json", dest="source_json", type=Path, default=None,
                       help="local MatPES json.gz; skips the download")
    build.add_argument("--atoms", dest="atoms_file", type=Path, default=None,
                       help="isolated-atom energies; looked for under --root and the cwd when omitted")
    build.add_argument("--cutoff", type=float, default=5.0, help="graph cutoff in A (released model: 5.0)")
    build.add_argument("--limit", type=int, default=None, help="first N records only (smoke tests)")

    folds.add_argument("--out-name", default="artifacts_full")
    folds.add_argument("--folds", type=int, default=3)
    folds.add_argument("--seed0", type=int, default=42, help="fold k uses seed0 + k")
    folds.add_argument("--train-frac", type=float, default=0.90)
    folds.add_argument("--val-frac", type=float, default=0.05)
    folds.add_argument("--cutoff", type=float, default=5.0)
    folds.add_argument("--threebody-cutoff", type=float, default=4.0)
    folds.add_argument("--functional", default="R2SCAN")
    folds.add_argument("--overwrite", action="store_true", help="replace existing fold directories")
    folds.add_argument("--element-refs", type=Path, default=None,
                       help="defaults to <root>/artifacts/element_refs.npy; isolated-atom "
                            "energies are split-independent, so they are copied, not refitted")
    args = p.parse_args(argv)

    cfg = DIEPConfig(root=args.root or default_root(), cache_name=args.cache_name)
    if args.command == "build":
        cfg.root.mkdir(parents=True, exist_ok=True)
        build_cache(cfg, functional=args.functional, cutoff=args.cutoff, source_json=args.source_json,
                    atoms_file=args.atoms_file, limit=args.limit)
        return 0

    refs = args.element_refs or (cfg.root / "artifacts" / "element_refs.npy")
    out_dir = cfg.root / args.out_name
    build_folds(cfg.cache_dir, out_dir, n_folds=args.folds, seed0=args.seed0,
                element_refs_src=refs, cutoff=args.cutoff,
                threebody_cutoff=args.threebody_cutoff, train_frac=args.train_frac,
                val_frac=args.val_frac, functional=args.functional, overwrite=args.overwrite)
    print(f"wrote {args.folds} folds under {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
