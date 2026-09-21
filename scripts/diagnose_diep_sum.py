#!/usr/bin/env python
"""Diagnostic: does a bond-length-gated DIEPIntegrator(mode="sum") give genuinely different
channels, or is it still a rank-1 bond descriptor in disguise?

Builds graphs for the sample CIFs in test_structures/, runs the integrator exactly as
the model does (same code path as DIEP.forward, minus everything downstream of g.rbf),
with a bank of Gaussian windows in bond-length space (NUM_CHANNELS centers, evenly spaced
between CENTER_MIN and CENTER_MAX, each of width CHANNEL_WIDTH) gating the single physics-
integral scalar. Reports the rank and cross-channel correlation of the resulting
(n_bonds, NUM_CHANNELS) feature matrix, and plots each channel's sorted values plus a
scatter against bond length.
"""

from __future__ import annotations

import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from pymatgen.core import Structure

import diep
from diep.config import DEFAULT_ELEMENTS
from diep.pyg.graph.compute import compute_pair_vector_and_distance
from diep.pyg.graph.converters import Structure2Graph
from diep.pyg.layers._diep import DIEPIntegrator

CUTOFF = 5.0

NUM_CHANNELS = 8
CENTER_MIN, CENTER_MAX = 1.5, 5.0
CHANNEL_WIDTH = 0.4
CENTERS = np.linspace(CENTER_MIN, CENTER_MAX, NUM_CHANNELS).tolist()

converter = Structure2Graph(element_types=DEFAULT_ELEMENTS, cutoff=CUTOFF)
integrator = DIEPIntegrator(mode="sum", channel_centers=CENTERS, channel_width=CHANNEL_WIDTH)

from pymatgen.core.periodic_table import Element

element_z = torch.tensor([Element(el).Z for el in DEFAULT_ELEMENTS], dtype=diep.float_th)

all_feat, all_dist, all_labels = [], [], []

cif_paths = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_structures", "*.cif")))
print(f"Found {len(cif_paths)} sample CIFs: {[os.path.basename(p) for p in cif_paths]}")

for path in cif_paths:
    name = os.path.basename(path)
    structure = Structure.from_file(path)
    graph, lattice, state_attr = converter.get_graph(structure)
    graph.pos = torch.tensor(structure.cart_coords, dtype=diep.float_th)
    graph.pbc_offshift = torch.matmul(graph.pbc_offset, lattice[0])
    bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
    graph.bond_vec = bond_vec
    graph.bond_dist = bond_dist

    atomic_numbers = element_z[graph.node_type.long()]

    with torch.no_grad():
        bond_feat, _ = integrator(graph, atomic_numbers, compute_triplets=False)

    feat = bond_feat.numpy()  # (n_bonds, NUM_CHANNELS)
    dist = bond_dist.numpy()
    print(f"{name}: {len(feat)} bonds, feat range [{feat.min():.6g}, {feat.max():.6g}], "
          f"dist range [{dist.min():.3f}, {dist.max():.3f}] A")

    all_feat.append(feat)
    all_dist.append(dist)
    all_labels.append(np.full(len(dist), name))

all_feat = np.concatenate(all_feat, axis=0)  # (N, NUM_CHANNELS)
all_dist = np.concatenate(all_dist)
all_labels = np.concatenate(all_labels)

print(f"\nTotal bonds across sample: {len(all_feat)}")
print(f"Overall feat range: [{all_feat.min():.6g}, {all_feat.max():.6g}]")

rank = np.linalg.matrix_rank(all_feat - all_feat.mean(axis=0, keepdims=True))
print(f"Rank of centered (N, {NUM_CHANNELS}) feature matrix: {rank} (out of {NUM_CHANNELS} channels)")

corr = np.corrcoef(all_feat, rowvar=False)
off_diag = corr[~np.eye(NUM_CHANNELS, dtype=bool)]
print(f"Cross-channel correlation: mean |r| = {np.abs(off_diag).mean():.4f}, "
      f"max |r| = {np.abs(off_diag).max():.4f} (1.0 would mean every channel is redundant)")
print(f"Channel centers (A): {[round(c, 3) for c in CENTERS]}, width = {CHANNEL_WIDTH}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for d in range(NUM_CHANNELS):
    axes[0].plot(np.sort(all_feat[:, d]), marker=".", markersize=2, linestyle="none",
                 label=f"center={CENTERS[d]:.2f}", alpha=0.7)
axes[0].set_xlabel("sorted bond index")
axes[0].set_ylabel("DIEP bond feature (g.rbf), per channel")
axes[0].set_title(f"Sorted bond_feat per center channel ({len(all_feat)} bonds, {len(cif_paths)} structures)")
axes[0].set_yscale("symlog")
axes[0].legend(fontsize=6, ncol=2)

for d in [0, NUM_CHANNELS // 2, NUM_CHANNELS - 1]:
    axes[1].scatter(all_dist, all_feat[:, d], s=6, alpha=0.5, label=f"center={CENTERS[d]:.2f}")
axes[1].set_xlabel("bond length (A)")
axes[1].set_ylabel("DIEP bond feature")
axes[1].set_yscale("symlog")
axes[1].set_title("bond_feat vs bond length, smallest/middle/largest center")
axes[1].legend(fontsize=7)

fig.tight_layout()
out_path = "/tmp/diep_sum_diagnosis.png"
fig.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
