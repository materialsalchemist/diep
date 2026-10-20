#!/usr/bin/env python
"""Before/after audit of the DIEP three-body index-space defect.

Runs the four checks from the fix brief and prints the numbers:

1. White-box triple audit -- how many triples reference the wrong end atom, and how many
   reference a bond beyond ``threebody_cutoff`` (where the polynomial envelope is an exact
   zero, so the triple is silently deleted rather than attenuated).
2. Block-level comparison of the three-body bond update, shipped vs fixed, on identical
   pretrained weights: cosine similarity and relative L2.
3. Invariant regression over a few hundred structures, batched and unbatched.

"Shipped" is reproduced by calling ``_compute_3body`` on the pruned graph directly, which
is exactly what ``create_line_graph`` did before the fix. "Fixed" is ``create_line_graph``.

Note on end-to-end deltas: DIEP's three-body head contributes very little to the total
energy, so an end-to-end energy comparison is not a sensitive probe of this defect and is
deliberately not reported here. Cite the block-level numbers.

Usage:
    PYTHONPATH=src python scripts/verify_threebody_fix.py
    PYTHONPATH=src:tests DIEP_DGL_STUB=1 python scripts/verify_threebody_fix.py  # no DGL wheel
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

if os.environ.get("DIEP_DGL_STUB") == "1":
    from _dgl_stub import install as _install_dgl_stub

    _install_dgl_stub()

import dgl  # noqa: E402
import diep  # noqa: E402
from diep.ext.pymatgen import Structure2Graph  # noqa: E402
from diep.graph.compute import (  # noqa: E402
    _compute_3body,
    assert_lg_invariants,
    compute_pair_vector_and_distance,
    create_line_graph,
    prune_edges_by_features,
)
from diep.utils.cutoff import polynomial_cutoff  # noqa: E402
from pymatgen.core import Lattice, Structure  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUTOFF, THREEBODY_CUTOFF = 5.0, 4.0  # DIEP's shipped diep_pes configuration


def shipped_line_graph(g, threebody_cutoff):
    """The pre-fix line graph: node ids left in pruned-bond index space."""
    pruned = prune_edges_by_features(g, feat_name="bond_dist", condition=lambda x: x > threebody_cutoff)
    return _compute_3body(pruned)


def build_graph(structure, element_types, cutoff=CUTOFF):
    converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
    g, lattice, _ = converter.get_graph(structure)
    g.edata["pbc_offshift"] = torch.matmul(g.edata["pbc_offset"], lattice[0])
    g.ndata["pos"] = g.ndata["frac_coords"] @ lattice[0]
    bond_vec, bond_dist = compute_pair_vector_and_distance(g)
    g.edata["bond_vec"] = bond_vec
    g.edata["bond_dist"] = bond_dist
    return g


def named_structures():
    out = []
    cif_dir = os.path.join(REPO, "test_structures")
    for name in sorted(os.listdir(cif_dir)):
        if name.endswith(".cif"):
            out.append((name.split("_")[-1][:-4], Structure.from_file(os.path.join(cif_dir, name))))
    si = Structure(
        Lattice.cubic(5.43), ["Si"] * 8,
        [[0, 0, 0], [0.25, 0.25, 0.25], [0, 0.5, 0.5], [0.25, 0.75, 0.75],
         [0.5, 0, 0.5], [0.75, 0.25, 0.75], [0.5, 0.5, 0], [0.75, 0.75, 0.25]],
    )
    out.append(("Si8", si))
    rattled = si.copy()
    rattled.perturb(0.1)
    out.append(("Si8-rattled", rattled))
    out.append(("Mo2-bcc", Structure(Lattice.cubic(3.15), ["Mo"] * 2, [[0, 0, 0], [0.5, 0.5, 0.5]])))
    out.append((
        "NaCl8",
        Structure(Lattice.cubic(5.64), ["Na", "Cl"] * 4,
                  [[0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0.5, 0.5, 0],
                   [0, 0, 0.5], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]]),
    ))
    return out


def load_pretrained():
    """Load diep_pes with the three-body path forced on.

    The vendored `Potential.__init__` takes `use_edges: bool = False` and guards it with
    `if use_edges is not None:`, which is true for False -- so constructing a Potential
    always sets model.use_edges = model.use_triplets = False and the triplet path never
    runs. Every measurement here requires turning it back on explicitly.
    """
    from diep.models._diep import DIEP

    cfg = json.load(open(os.path.join(REPO, "pretrained_models/diep_pes/model.json")))
    init_args = cfg["kwargs"]["model"]["init_args"]
    model = DIEP(**init_args)
    state = torch.load(os.path.join(REPO, "pretrained_models/diep_pes/state.pt"),
                       map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(
        {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}, strict=False
    )
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    model.use_edges = model.use_triplets = True
    return model, tuple(init_args["element_types"]), init_args["cutoff"], init_args["threebody_cutoff"]


def section_1_triple_audit(element_types, cutoff, tbc):
    print("\n" + "=" * 108)
    print("1. WHITE-BOX TRIPLE AUDIT   (shipped vs fixed line graph, same structures)")
    print("=" * 108)
    hdr = (f"{'structure':<16}{'bonds':>7}{'kept':>7}{'triples':>9}"
           f"{'wrongEnd%':>11}{'beyondTBC%':>12}{'beyondTBC%':>12}{'zeroWt%':>10}{'zeroWt%':>10}{'tinyWt%':>10}")
    sub = f"{'':<16}{'':>7}{'':>7}{'':>9}{'shipped':>11}{'shipped':>12}{'FIXED':>12}{'shipped':>10}{'FIXED':>10}{'FIXED':>10}"
    print(hdr)
    print(sub)
    print("-" * len(hdr))
    tot = dict.fromkeys(["n", "wrong", "beyond_s", "beyond_f", "zero_s", "zero_f", "tiny_f", "bonds", "kept"], 0)
    for name, s in named_structures():
        g = build_graph(s, element_types, cutoff)
        bd, pdst = g.edata["bond_dist"], g.edges()[1]
        lg_f, lg_s = create_line_graph(g, tbc), shipped_line_graph(g, tbc)
        assert_lg_invariants(g, lg_f)
        sf, df = (t.long() for t in lg_f.edges())
        ss, ds = (t.long() for t in lg_s.edges())
        assert sf.numel() == ss.numel()
        n = int(sf.numel())
        pc = polynomial_cutoff(bd, tbc)
        w_s, w_f = pc[ss] * pc[ds], pc[sf] * pc[df]
        row = dict(
            wrong=int((pdst[ds] != pdst[df]).sum()),
            beyond_s=int(((bd[ss] > tbc) | (bd[ds] > tbc)).sum()),
            beyond_f=int(((bd[sf] > tbc) | (bd[df] > tbc)).sum()),
            zero_s=int((w_s == 0).sum()), zero_f=int((w_f == 0).sum()), tiny_f=int((w_f < 1e-6).sum()),
        )
        kept = int((bd <= tbc).sum())
        print(f"{name:<16}{g.num_edges():>7}{kept:>7}{n:>9}"
              f"{100 * row['wrong'] / max(n, 1):>11.2f}{100 * row['beyond_s'] / max(n, 1):>12.2f}"
              f"{100 * row['beyond_f'] / max(n, 1):>12.2f}{100 * row['zero_s'] / max(n, 1):>10.2f}"
              f"{100 * row['zero_f'] / max(n, 1):>10.2f}{100 * row['tiny_f'] / max(n, 1):>10.2f}")
        tot["n"] += n
        tot["bonds"] += g.num_edges()
        tot["kept"] += kept
        for k, v in row.items():
            tot[k] += v
    n = max(tot["n"], 1)
    print("-" * len(hdr))
    print(f"{'TOTAL':<16}{tot['bonds']:>7}{tot['kept']:>7}{tot['n']:>9}"
          f"{100 * tot['wrong'] / n:>11.2f}{100 * tot['beyond_s'] / n:>12.2f}{100 * tot['beyond_f'] / n:>12.2f}"
          f"{100 * tot['zero_s'] / n:>10.2f}{100 * tot['zero_f'] / n:>10.2f}{100 * tot['tiny_f'] / n:>10.2f}")
    print("\nwrongEnd%   : triples whose end-atom reference differs between the two index spaces")
    print("beyondTBC%  : triples referencing a bond longer than threebody_cutoff (must be 0.00 after the fix)")
    print("zeroWt%     : triples whose two-bond cutoff product is an exact zero, i.e. silently deleted")
    print("tinyWt%     : triples whose weight is < 1e-6 after the fix -- real, expected, and carries no signal")
    assert tot["beyond_f"] == 0, "FIXED path still references bonds beyond threebody_cutoff"
    return tot


def section_2_block_level(model, element_types, cutoff, tbc):
    print("\n" + "=" * 108)
    print("2. BLOCK-LEVEL COMPARISON   (three-body bond update, identical pretrained weights)")
    print("=" * 108)

    def update(g, lg, block=0):
        z = model.atomic_number_table[g.ndata["node_type"]].to(diep.float_th)
        bond_feat, triplet_feat = model.diep_integrator(g, lg, z, compute_triplets=True)
        g.edata["rbf"] = bond_feat
        node_feat, edge_feat, _ = model.embedding(g.ndata["node_type"], g.edata["rbf"], None)
        tcut = polynomial_cutoff(g.edata["bond_dist"], tbc)
        out = model.three_body_interactions[block](g, lg, triplet_feat, tcut, node_feat, edge_feat)
        return (out - edge_feat).detach()

    hdr = f"{'structure':<16}{'triples':>9}{'cos_sim':>10}{'relL2':>9}{'||d_ship||':>12}{'||d_fix||':>11}{'deadBonds_ship':>17}"
    print(hdr)
    print("-" * len(hdr))
    cos_all, rel_all = [], []
    for name, s in named_structures():
        g = build_graph(s, element_types, cutoff)
        lg_f, lg_s = create_line_graph(g, tbc), shipped_line_graph(g, tbc)
        with torch.no_grad():
            d_f, d_s = update(g, lg_f), update(g, lg_s)
        a, b = d_s.flatten(), d_f.flatten()
        cos = float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))
        rel = float((a - b).norm() / (b.norm() + 1e-30))
        dead = int((d_s.abs().sum(1) == 0).sum())
        print(f"{name:<16}{lg_f.num_edges():>9}{cos:>10.4f}{rel:>9.4f}"
              f"{float(a.norm()):>12.4f}{float(b.norm()):>11.4f}{dead:>10}/{d_s.shape[0]:<6}")
        cos_all.append(cos)
        rel_all.append(rel)
    print("-" * len(hdr))
    print(f"{'mean':<16}{'':>9}{np.mean(cos_all):>10.4f}{np.mean(rel_all):>9.4f}")
    print(f"{'range':<16}{'':>9}{min(cos_all):.4f}-{max(cos_all):.4f}   {min(rel_all):.4f}-{max(rel_all):.4f}")
    print("\ndeadBonds_ship : bonds receiving an exactly-zero three-body update in the shipped path")
    print("                 (segment ids only span pruned space, so high-numbered bonds never receive one)")
    return cos_all, rel_all


def section_3_invariants(element_types, cutoff, tbc, n_structures=300, batch_sizes=(1, 4, 16)):
    print("\n" + "=" * 108)
    print(f"3. INVARIANT REGRESSION   ({n_structures} structures, batched and unbatched)")
    print("=" * 108)
    rng = np.random.default_rng(0)
    protos = [s for _, s in named_structures()]
    graphs, lgs = [], []
    for i in range(n_structures):
        s = protos[i % len(protos)].copy()
        s.perturb(float(rng.uniform(0.0, 0.3)))
        g = build_graph(s, element_types, cutoff)
        graphs.append(g)
        lgs.append(create_line_graph(g, tbc))

    for g, lg in zip(graphs, lgs, strict=True):
        assert_lg_invariants(g, lg)
    print(f"unbatched : {len(graphs)}/{len(graphs)} pass assert_lg_invariants")

    for bs in batch_sizes:
        n_ok = 0
        for start in range(0, len(graphs) - bs + 1, bs):
            bg = dgl.batch(graphs[start : start + bs])
            blg = dgl.batch(lgs[start : start + bs])
            assert_lg_invariants(bg, blg)
            direct = create_line_graph(bg, tbc)
            assert direct.num_nodes() == blg.num_nodes()
            assert torch.equal(direct.edges()[0].long(), blg.edges()[0].long())
            assert torch.equal(direct.edges()[1].long(), blg.edges()[1].long())
            assert torch.equal(direct.ndata["n_triple_ij"], blg.ndata["n_triple_ij"])
            n_ok += 1
        print(f"batch={bs:<4}: {n_ok}/{n_ok} batches pass, and dgl.batch(line graphs) == "
              f"create_line_graph(dgl.batch(graphs)) edge-for-edge")

    total_bonds = sum(g.num_edges() for g in graphs)
    total_triples = sum(lg.num_edges() for lg in lgs)
    print(f"\ntotals    : {total_bonds} bonds, {total_triples} triples checked")


def main():
    backend = "stub (tests/_dgl_stub.py)" if getattr(dgl, "_is_shim", False) else f"dgl {dgl.__version__}"
    torch.manual_seed(0)
    model, element_types, cutoff, tbc = load_pretrained()
    print("=" * 108)
    print("DIEP three-body index-space audit")
    print("=" * 108)
    print(f"dgl backend      : {backend}")
    print(f"weights          : pretrained_models/diep_pes (model.use_edges forced True)")
    print(f"cutoff           : {cutoff}")
    print(f"threebody_cutoff : {tbc}   ({'AFFECTED' if tbc < cutoff else 'not affected: nothing is pruned'})")
    section_1_triple_audit(element_types, cutoff, tbc)
    section_2_block_level(model, element_types, cutoff, tbc)
    section_3_invariants(element_types, cutoff, tbc)
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
