# DIEP-MatPES: a DIEP interatomic potential trained on MatPES r2SCAN

A pretrained machine-learned interatomic potential, `diep_fold1`, and all of the code that
trained it. The model is DIEP (direct integration of the external potential; Tawfik *et
al.*, Digital Discovery 2024) on PyTorch Geometric. It was trained on 349,107 structures
from the MatPES r2SCAN dataset and predicts energies, forces and stresses for 89 elements.
It runs as an ASE calculator on CPU or GPU and needs neither DGL nor matgl.

| fold-1 test split (19,395 structures) | MAE | RMSE* |
|---|---|---|
| Energy | **26.2 meV/atom** | 46.4 meV/atom |
| Forces | **0.144 eV/Å** | 0.291 eV/Å |
| Stress | **0.646 GPa** | 1.21 GPa |

\* As logged during training: the mean of per-batch RMSEs, not the RMSE over the split,
and it understates the latter: over the first 640 test structures the energy RMSE is
54 meV/atom, against a logged 44. `diep_pyg.evaluate` reports both (see
[Evaluating](#evaluating)).

The checkpoint is from epoch 295 of 300, chosen on validation loss (0.2317). Training ran
the full cosine schedule and did not early-stop.

## Contents

```
models/diep_fold1/            the pretrained model
    model.json model.pt state.pt   exported Potential (load with diep_pyg.pretrained)
    PROVENANCE.json                source checkpoint, md5, architecture, metrics
    checkpoint/best-epoch=0295.ckpt   the Lightning checkpoint (optimizer state included)
    fold1/                         splits.json, meta.json, element_refs.npy of its split
    training/                      CSV logs, merged log, curves.png, metrics.json
diep_pyg/                     DIEP on PyTorch Geometric, and the code that trained this model:
    matpes.py                      MatPES -> graph cache, fold splits, data loaders
    train.py                       the training script
    export.py evaluate.py          checkpoint -> model directory; metrics on a split
    pretrained.py                  load_potential / make_calculator
slurm/train.sh                the exact training command for diep_fold1
slurm/evaluate.sh             full test-split evaluation on a GPU
examples/predict.py           single point, optional relaxation
examples/md.py                NVT molecular dynamics
tools/training_curves.py      merge resumed CSV logs and plot curves
tests/                        105 tests (library + release)
environment.yml requirements.txt pyproject.toml
```

## Install

```bash
conda env create -f environment.yml
conda activate diep-matpes
pip install -e .                    # optional; scripts also run from this directory
pytest -q                           # ~30 s on CPU
```

Pinned: Python 3.11, torch 2.5.1, torch-geometric 2.7.0, lightning 2.6.1, pymatgen 2026.5,
ase 3.28. See `environment.yml` for GPU notes.

## Using the pretrained model

```python
from ase.build import bulk
from diep_pyg.pretrained import make_calculator

atoms = bulk("Si", "diamond", a=5.43, cubic=True)
atoms.calc = make_calculator()               # models/diep_fold1 on CPU
# atoms.calc = make_calculator(device="cuda", stress=False)

atoms.get_potential_energy()                 # eV
atoms.get_forces()                           # eV/Å
atoms.get_stress()                           # eV/Å³ (ASE convention)
```

```bash
python examples/predict.py my.cif --relax    # single point, then FIRE relaxation of cell + positions
python examples/md.py --temperature 1000 --steps 5000 --device cuda
```

- `make_calculator` returns a `DIEPCalculator`. It wraps positions into the cell before
  building the graph, which keeps the neighbour search fast in long MD runs. It reports
  stress in eV/Å³. The underlying `diep_pyg.ext.ase.PESCalculator` defaults to GPa,
  inherited from matgl, which ASE would read as 160× too large.
- `load_potential()` returns the bare `Potential` (energies in eV, forces in eV/Å, stresses
  in GPa) for batched use on PyG graphs.
- Loading is strict. A missing parameter raises an error rather than being left at its
  random initialisation.
- Diamond Si relaxes to a = 5.439 Å; experiment gives 5.431 Å.

## Training from scratch

The released model is reproduced by these four steps, with fold 1:

```bash
export DIEP_MATPES_ROOT=$PWD/data

# 1. MatPES r2SCAN -> graph cache + isolated-atom element references (holds the whole set in RAM)
python -m diep_pyg.matpes build --json MatPES-R2SCAN-2025.1.json.gz --atoms MatPES-R2SCAN-atoms.json.gz
#    (or `pip install matpes` and omit --json/--atoms to download)

# 2. seeded 90/5/5 splits; fold k uses seed 42+k            (~1 min)
python -m diep_pyg.matpes folds --folds 3

# 3. train fold 1 exactly as diep_fold1 was trained          (~76 GPU-hours on an L40S)
sbatch slurm/train.sh            # or: FOLD=1 bash slurm/train.sh

# 4. export the best checkpoint to a model directory
python -m diep_pyg.export \
    --ckpt data/runs_diep/diep_fold1/checkpoints/best-epoch=*.ckpt \
    --artifacts data/artifacts_full/fold1 --metrics data/runs_diep/diep_fold1/metrics.json \
    --out models/my_diep
```

**Hyperparameters** (all in `slurm/train.sh`):

| setting | value |
|---|---|
| graph / three-body cutoff | 5.0 / 4.0 Å |
| blocks, width | 3 conv blocks, 64 units; 64-dim node and edge embeddings |
| DIEP integrator | grid mode, half-length 5.0, spacing 1.0, Gaussian σ 1.0, softening 0.5, √Z effective charge |
| triplet frame | `canonical` (see [Known limitations](#known-limitations)) |
| parameters | 403,181 |
| loss | L1; weights energy (per atom) 1.0, force 1.0, stress 0.1 |
| optimiser | AdamW (amsgrad), lr 1e-3, weight decay 1e-5, gradient clip 2.0 |
| schedule | cosine over 300 epochs to 1e-5, stepped once per epoch |
| batch | 32 structures, gradient accumulation 4 |
| early stopping | patience 30 on validation total loss (not triggered) |
| energy scale | `data_std` = training-split RMS force norm, 3.1838 eV/Å |
| element refs | MatPES r2SCAN isolated-atom energies |

**Hardware.** The GPU needs at least 44 GB. Batches have no size cap, and the two largest
MatPES cells (216 atoms, about 574k triplets each) need about 23 GB on their own, so a 32 GB
V100 runs out of memory in about 5% of epochs. Peak host memory was 46 GiB (fold 1), and
32 GB was not enough.

**Resuming.** A run resumes from `checkpoints/last.ckpt` automatically. That file is
rewritten only when validation loss improves, so a resume restarts from the best epoch so
far. Early-stopping patience is restored from the checkpoint, and `--patience` is ignored
on resume. To fine-tune from the released weights instead, use
`--init-from models/diep_fold1/checkpoint/best-epoch=0295.ckpt`. That flag loads the
weights only and starts a fresh optimiser and schedule.

**How diep_fold1 was actually trained.** The run was interrupted twice, and each time it
resumed from its best checkpoint:

| job | GPU | epochs | ended |
|---|---|---|---|
| 1 | V100 32 GB | 0 | CUDA out of memory in epoch 1 |
| 2 | L40S | 1–72 | host out of memory (32 GB limit); best was 62 |
| 3 | L40S | 63–299, then test | completed, 60.1 h |

`training/metrics_merged.csv` keeps the later job for epochs 63–72, which were trained
twice. In total this was about 79 GPU-hours. Every job ran the `diep_pyg` code shipped here.
The one exception: jobs 1–2 predate the opt-in `triplet_frame="bond"` option, and adding
it left the canonical path that this model uses unchanged.

## Evaluating

```bash
python -m diep_pyg.evaluate --root data --predictions test_preds.npz   # or: sbatch slurm/evaluate.sh
```

This evaluates the released model on its own fold-1 test split, read from
`models/diep_fold1/fold1`, using the graph cache from step 1.

- **Metrics:** energy MAE in eV/atom, plus force and stress MAEs. Two RMSEs are reported.
  `*_RMSE_logged` reproduces the training log, while `*_RMSE` is taken over the whole split.
- **Per-structure output:** `--predictions` writes predicted and DFT energies per atom,
  force MAE, the largest force error and largest predicted force, and both stress tensors.

## Verification

Each item below was run while this release was assembled. The tests in `tests/` re-check
the parts that do not need the full dataset.

- **The export is the trained model.** All 113 tensors in `models/diep_fold1` equal the
  checkpoint's exactly. Energies, forces and stresses on three probe cells are
  bit-identical. This was checked before writing and again after reading the files back
  (`tests/test_release.py` repeats it).
- **The training code is the code that trained it.** `diep_pyg/train.py` and the
  original script each ran one epoch on the real MatPES cache, single-threaded (CPU, fold 1,
  64 training structures). The trained weights, the logged metrics and the test metrics
  were bit-identical.
- **The splits regenerate exactly.** `build_folds` on the training cache reproduces
  `splits.json`, `meta.json` and `element_refs.npy` byte for byte, for all three folds.
- **The graph cache regenerates.** `build_cache` reproduces the training cache field for field:
  graphs, dtypes, lattices, labels and element references. On a random 20,384-structure
  sample, 99.7% of graphs are bit-identical. The other 0.3% contain the same edges in a
  different order, and matgl's own converter run today reorders them the same way. The
  cause is the shared neighbour search, which has changed since the cache was built on
  2026-08-06. Edge order leaves the graph unchanged and affects only float32 summation
  order.
- **Evaluation matches training's metrics.** On the first 640 fold-1 test structures,
  `evaluate` reproduces Lightning's test metrics (MAEs and logged RMSEs) to within 2e-7
  relative.
- **Derivatives are consistent.** On rattled Si, MgO and NaCl, forces match central finite
  differences of the energy to 4e-3 eV/Å (float32, h = 0.01 Å). On Si, stresses match
  strain finite differences to 7e-5 eV/Å³.

## Known limitations

**The canonical triplet frame makes the energy slightly discontinuous.** Each three-body
term is drawn in a frame chosen by ordering the triplet's edges by length. Two edges of
different elements can cross in length. When they do, the order flips and the energy
steps: by 1.34 meV for an O–Ir–W triplet as the O–W bond crosses O–Ir at 2.0 Å (measured on
the fold-0 model trained the same way). Autograd forces are hardly affected, but two things
follow:

- NVE energy is not exactly conserved.
- Finite-difference checks fail near such crossings. In an earlier test of this model on
  liquid water, one H atom's finite-difference force was off by 0.12 eV/Å.

`diep_pyg/README.md` ("Triplet ordering", "Bond-anchored triplet frame") has the details.
For new training, `TRIPLET_FRAME=bond sbatch slurm/train.sh` selects a frame anchored to
each bond, which has no such steps. In a matched fold-0 comparison it also reached a lower
test energy MAE (31.8 vs 35.4 meV/atom). This model is canonical and must be evaluated
canonically. The frame is recorded in `model.json`, and bond-frame state dicts carry a
marker, so a strict load across frames fails instead of silently computing the wrong
features.

**Float32 force spikes on nearly straight triplets.** The canonical frame normalises
`perp/|perp|`, which amplifies float32 rounding when three atoms are nearly collinear and
off-centre. With the fold-0 model, spurious forces of up to 65–75 eV/Å appeared on 1.2% of
MatPES structures; float64 gives about 0.3 eV/Å on the same structures. Stress
is unaffected. In MD, watch the largest force: `examples/md.py` logs it and flags anything
above 10 eV/Å, and `evaluate` records it per structure. The bond frame removes this by
construction.

**Rare elements.** All 89 element types occur in the training split. The noble gases are
nearly absent: Ne appears in 1 training structure, Ar in 3, Kr in 23, He in 76 and Xe in
156. Treat predictions involving them with suspicion.

**Smaller items.**

- `diep_pyg.ext.ase.MolecularDynamics.set_atoms` keeps the integrator's cached masses. If
  you swap in a different system, build a new driver.
- `PESCalculator` mishandles the default `state_attr` for models with `include_state=True`.
  This model does not use state features.
- Training with `allow_missing_labels=True` can produce a NaN validation loss when a whole
  batch lacks one label type. The released model was trained without it.

## Citing

- DIEP: S. A. Tawfik, T. M. Nguyen, S. P. Russo, T. Tran, S. Gupta, S. Venkatesh,
  "Embedding material graphs using the electron-ion potential: application to material
  fracture", *Digital Discovery* (2024), https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00246f
- MatPES: A. D. Kaplan *et al.*, "A foundational potential energy surface dataset for
  materials", arXiv:2503.04070 (2025).

## License

BSD 3-Clause. `LICENSE` is the upstream `diep` repository's license file, carried over
unchanged. `diep_pyg` derives from that code, which in turn derives from matgl.
