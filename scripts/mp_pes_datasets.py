#!/usr/bin/env python
"""Download and parse Materials Project PES datasets for DIEP training.

Two dataset families are supported, both distributed under names conventionally
called the "MP PES dataset":

* MPF.2021.2.8 -- the M3GNet-era Materials Project PES dataset (figshare item
  19470599): ~190k structures with real DFT energies/forces/stresses, shipped as
  two pickled blocks. This is exactly what ``src/dataset.py::get_mp_pes_dataset``
  already parses -- the only thing missing there was the download step.
* MatPES-PBE-2025.2 -- from matpes.ai, distributed as a JSONL file on the Hugging
  Face Hub (repo ``materialyze/matpes``, confirmed against the upstream
  ``materialyzeai/matpes`` package's ``src/matpes/data.py``).

Each loader returns ``(structures, labels)`` where ``structures`` is a list of
``pymatgen.core.Structure`` and ``labels`` is ``{"energies": [...], "forces": [...],
"stresses": [...]}`` -- the format ``diep.pyg.graph.data.DIEPDataset`` expects.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import requests
from ase.stress import voigt_6_to_full_3x3_stress
from pymatgen.core import Structure
from tqdm import tqdm

FIGSHARE_ARTICLE_URL = "https://api.figshare.com/v2/articles/19470599"
MATPES_REPO_ID = "materialyze/matpes"


def download_mpf_2021(dest_dir: str) -> list[str]:
    """Download the two MPF.2021.2.8 pickle blocks from figshare, if not already present.

    Args:
        dest_dir: root directory to download into. Files land at
            ``<dest_dir>/MPF.2021.2.8/19470599/<name>``, mirroring the layout the
            original (non-downloading) ``src/dataset.py`` loader already assumed.

    Returns:
        Local paths to ``block_0.p`` and ``block_1.p``.
    """
    article = requests.get(FIGSHARE_ARTICLE_URL, timeout=30).json()
    out_dir = os.path.join(dest_dir, "MPF.2021.2.8", "19470599")
    os.makedirs(out_dir, exist_ok=True)

    paths = []
    for f in article["files"]:
        path = os.path.join(out_dir, f["name"])
        if os.path.exists(path) and os.path.getsize(path) == f["size"]:
            print(f"Already downloaded: {path}")
            paths.append(path)
            continue
        print(f"Downloading {f['name']} ({f['size'] / 1e6:.1f} MB) from {f['download_url']}")
        with requests.get(f["download_url"], stream=True, timeout=30) as r:
            r.raise_for_status()
            with open(path, "wb") as fh, tqdm(total=f["size"], unit="B", unit_scale=True, desc=f["name"]) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
                    bar.update(len(chunk))
        paths.append(path)
    return paths


def download_matpes(cache_dir: str, functional: str = "PBE", version: str = "2025.2") -> str:
    """Download a MatPES JSONL file from the Hugging Face Hub, if not already cached.

    Args:
        cache_dir: Hugging Face Hub cache directory to use.
        functional: "PBE" or "R2SCAN".
        version: dataset version string.

    Returns:
        Local path to the downloaded ``.jsonl`` file.
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=MATPES_REPO_ID,
        filename=f"MatPES-{functional.upper()}-{version}.jsonl",
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def load_mpf_2021(
    block_paths: list[str],
    max_structures: int | None = None,
    force_limit: float | None = None,
) -> tuple[list[Structure], dict[str, list]]:
    """Parse the MPF.2021.2.8 pickle blocks into structures + PES labels.

    Args:
        block_paths: paths to the pickled blocks, as returned by :func:`download_mpf_2021`.
        max_structures: stop after this many structures (``None`` = no limit).
        force_limit: skip structures whose max absolute force component exceeds this
            value (``None`` = no filtering).

    Returns:
        (structures, labels)
    """
    data: dict = {}
    for path in block_paths:
        with open(path, "rb") as f:
            data.update(pickle.load(f))

    structures, energies, forces, stresses = [], [], [], []
    for item in data.values():
        for iid in range(len(item["energy"])):
            force = np.array(item["force"][iid])
            if force_limit is not None and np.abs(force).max() > force_limit:
                continue
            structures.append(item["structure"][iid])
            energies.append(item["energy"][iid])
            forces.append(force.tolist())
            stresses.append(np.array(item["stress"][iid]).tolist())
            if max_structures is not None and len(structures) >= max_structures:
                break
        if max_structures is not None and len(structures) >= max_structures:
            break

    print(f"MPF.2021.2.8: loaded {len(structures)} structures")
    return structures, {"energies": energies, "forces": forces, "stresses": stresses}


def load_matpes(jsonl_path: str, max_structures: int | None = None) -> tuple[list[Structure], dict[str, list]]:
    """Parse a MatPES JSONL file into structures + PES labels.

    Stress in MatPES records is in kbar, VASP compressive-positive convention; it is
    converted here to GPa, compressive-negative (the pymatgen/ASE convention), using the
    standard ``voigt_6_to_full_3x3_stress(...) * -0.1`` factor for this dataset family.

    Args:
        jsonl_path: path to the ``.jsonl`` file, as returned by :func:`download_matpes`.
        max_structures: stop after this many structures (``None`` = no limit).

    Returns:
        (structures, labels)
    """
    structures, energies, forces, stresses = [], [], [], []
    with open(jsonl_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            structures.append(Structure.from_dict(record["structure"]))
            energies.append(record["energy"])
            forces.append(record["forces"])
            stresses.append((voigt_6_to_full_3x3_stress(np.array(record["stress"])) * -0.1).tolist())
            if max_structures is not None and len(structures) >= max_structures:
                break

    print(f"MatPES: loaded {len(structures)} structures")
    return structures, {"energies": energies, "forces": forces, "stresses": stresses}


def load_datasets(
    data_dir: str,
    datasets: str = "both",
    matpes_functional: str = "PBE",
    matpes_version: str = "2025.2",
    max_structures: int | None = None,
    force_limit: float | None = None,
) -> tuple[list[Structure], dict[str, list]]:
    """Download (if needed) and load the requested dataset(s), concatenated together.

    Args:
        data_dir: root directory for downloads/caches.
        datasets: "mpf", "matpes" or "both".
        matpes_functional: "PBE" or "R2SCAN".
        matpes_version: MatPES dataset version.
        max_structures: per-dataset cap on the number of structures loaded.
        force_limit: MPF-only outlier filter (see :func:`load_mpf_2021`).

    Returns:
        (structures, labels), combined across the selected dataset(s).
    """
    structures: list[Structure] = []
    labels: dict[str, list] = {"energies": [], "forces": [], "stresses": []}

    if datasets in ("mpf", "both"):
        block_paths = download_mpf_2021(data_dir)
        s, lb = load_mpf_2021(block_paths, max_structures=max_structures, force_limit=force_limit)
        structures += s
        for k in labels:
            labels[k] += lb[k]

    if datasets in ("matpes", "both"):
        jsonl_path = download_matpes(data_dir, functional=matpes_functional, version=matpes_version)
        s, lb = load_matpes(jsonl_path, max_structures=max_structures)
        structures += s
        for k in labels:
            labels[k] += lb[k]

    print(f"Combined dataset: {len(structures)} structures")
    return structures, labels


def _cli():
    parser = argparse.ArgumentParser(description="Download the MP PES training datasets")
    parser.add_argument("--data-dir", default="data/mp_pes")
    parser.add_argument("--datasets", choices=["mpf", "matpes", "both"], default="both")
    parser.add_argument("--matpes-functional", default="PBE")
    parser.add_argument("--matpes-version", default="2025.2")
    args = parser.parse_args()

    if args.datasets in ("mpf", "both"):
        download_mpf_2021(args.data_dir)
    if args.datasets in ("matpes", "both"):
        path = download_matpes(args.data_dir, functional=args.matpes_functional, version=args.matpes_version)
        print(f"MatPES downloaded to: {path}")


if __name__ == "__main__":
    _cli()
