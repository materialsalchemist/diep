# diep_pyg

DIEP on PyTorch Geometric — a standalone package with no parent and no DGL dependency.
Everything it needs is vendored here, including the framework-neutral pieces (MLP blocks,
activations, cutoffs, grid geometry, checkpoint IO) that used to come from the DGL-era
`diep`. Parameter names still match that implementation, so compatible checkpoints load
directly. The environment is pinned in `environment.yml` at the repository root, and the
tests named below live in the top-level `tests/` directory. This file is the engineering
record of the port: what was fixed, how it was verified, and why.

## Layout

| path | role |
| --- | --- |
| `graph/compute.py` | bond vectors, edge pruning, the three-body line graph |
| `graph/converters.py` | pymatgen `Structure` / `Molecule` → `DIEPData` |
| `graph/data.py` | dataset, collate functions, dataloaders |
| `layers/` | embedding, three-body, graph convolution, readouts, ZBL |
| `layers/_diep.py` | the DIEP electron-ion potential integrator |
| `layers/_diep_core.py` | grid, Gaussian density, local frames, triplet canonicalisation |
| `models/_diep.py` | the DIEP model |
| `apps/pes.py` | `Potential` (energies, forces, stresses, hessian) |
| `ext/ase.py` | ASE calculator, optimisers, MD drivers |
| `utils/training.py` | Lightning modules |
| `matpes.py` | MatPES → graph cache, fold splits, train/val/test loaders |
| `train.py` | the training script that produced `diep_fold1` (`python -m diep_pyg.train`) |
| `export.py` | training checkpoint → self-contained model directory |
| `evaluate.py` | metrics and per-structure predictions on a split |
| `pretrained.py` | `load_potential` / `make_calculator` for an exported model |

## Index spaces

The three-body line graph is built directly in **parent-bond index space**: `triple_index`
holds ids of bonds in the full graph, never of the pruned subset. That is the invariant the
DGL side had to reach by remapping, and `graph.compute.assert_lg_invariants` checks it
here. This is why the package does not inherit the M3GNet three-body index defect.

### Testing this invariant: fixtures must be asymmetric (2026-09-29, fifth pass)

`tests/test_parent_bond_index_space.py` pins the invariant under batching. It exists because
the batching coverage that preceded it **could not have detected a violation**.

`DIEPData.__inc__` must offset `triple_index` by the running *bond* count. Offsetting by the
*node* count is precisely the M3GNet defect. A test batching structures whose node count
equals their edge count cannot tell the two apart -- both offsets are the same number, so the
test passes either way. Every fixture in `test_batching_regressions.py` is that shape (diamond
Si is 2 atoms; the lone atom is 1 atom / 0 edges): those tests pin the readout and state
defects they were written for, and say nothing about node-vs-bond offsets.

Confirmed by mutation: with `__inc__` returning `self.num_nodes`, the **entire pre-existing
71-test suite still passes**, while 9 of the 12 new tests fail. The fixtures here are
deliberately asymmetric -- `SPARSE` has 5 nodes and 2 edges -- and
`test_fixtures_are_asymmetric` guards that property, so a later edit that "tidies" them into
a symmetric cell fails loudly instead of silently disarming the file.

The strongest check in it is `test_batching_prebuilt_line_graphs_equals_building_from_the_batch`:
batching per-structure line graphs must give exactly what enumerating triples over the
assembled batch gives. Those are two independent routes to the same object, and a wrong offset
breaks only the first. The end-to-end pair then shows the symptom as a *number* -- batched
energies equal per-structure energies, in either batch order -- since cross-structure triples
change the energy and nothing else in the pipeline would complain.

Anything added here that claims to pin an index space should be mutation-tested the same way.
A test of this invariant that has never been seen to fail is not evidence.

## Differences from the DGL backend

* The PyG model applies two-body cutoff smoothing, so predictions differ from the
  unsmoothed DGL backend even with identical weights.
* `apps.pes.Potential.use_edges` defaults to `None` and is applied only when not `None`.
  The DGL version defaults it to `False` and guards with `if use_edges is not None:`, which
  is true for `False` — so constructing a DGL `Potential` always switches the triplet path
  off, including inside `PotentialLightningModule`. Here, leaving it unset keeps whatever
  the model was built with.
* There is no `l_g` in a batch; the line graph travels on the graph itself as
  `triple_index` / `n_triple_ij`.

## Batched-path fixes (2026-09-28)

Two defects that only appeared once more than one structure was in a batch. Both were
silent on the happy path — a periodic cell with bonds throughout and no state features —
and neither is caught by the single-structure numerical checks below.
`tests/test_batching_regressions.py` covers both; each test was confirmed to fail with its
fix reverted.

* `layers/_embedding.py` unsqueezed the continuous state vector unconditionally, turning a
  batch's `(B, F)` into `(1, B, F)`. The embedding then emitted one state row regardless of
  batch size and the conv block's `state_feat[data.batch[src]]` indexed past the end, so
  `include_state=True` with `dim_state_feats` raised `IndexError` for every batch larger
  than one structure — i.e. it could not be trained. The `ntypes_state` / `nn.Embedding`
  branch was never affected. Now only a bare `(F,)` vector gains the leading axis.
* The `field="edge_feat"` readouts in `layers/_readout.py` sized their output from
  `int(edge_batch.max()) + 1`. A structure with no bonds contributes no entries to the edge
  batch vector, so a batch whose **last** structure was edgeless (an isolated atom, or any
  fragment with no neighbour inside the cutoff) produced one row too few — a silent
  misalignment of predictions against labels, not an error. Position matters: an edgeless
  structure anywhere but the tail leaves a larger index behind it, so the count came out
  right and the defect hid. All readouts now take the count from `_num_graphs`, which reads
  the node assignment; every structure has at least one atom. The node-indexed sites
  (`_atom_ref`, `_zbl`, `WeightedAtomReadOut`, the extensive branch of `models/_diep.py`)
  were correct already and are unchanged in behaviour.

## Derivative-flag and checkpoint fixes (2026-09-28, second pass)

Two defects found by a second review. Neither appears on the default configuration --
`calc_forces=True` with an honest checkpoint -- which is why the numerical and batching
checks above all passed while they were live.
`tests/test_derivative_flag_regressions.py` covers both; each test was confirmed to fail
with its fix reverted.

* `apps/pes.py` computed the energy gradient only under `calc_forces`, then let the
  `calc_hessian`, `calc_stresses` and `debug_mode` branches index the result
  unconditionally. Asking for a **Hessian or a stress without forces** raised `TypeError`
  on `None`, and `debug_mode` without stresses raised `IndexError` off the end of a
  one-element list -- so energy-only Hessians and stress-without-force evaluation were
  unreachable. `g.pos` was separately only made differentiable under `calc_forces`, so the
  Hessian path had no graph to differentiate even once the indexing was corrected. Both
  gradients are now collected by name and requested whenever any consumer needs them; no
  branch indexes a position that may not have been filled. All eight flag combinations run,
  and the stress and Hessian are bit-identical whether or not forces are also requested.

* `utils/io.py` loaded weights with `load_state_dict(..., strict=False)`. A checkpoint whose
  keys did not line up loaded with no exception, no warning and no change in output shape,
  leaving every unmatched parameter at its **random initialisation** -- the model then
  predicted confidently from untrained weights. This is the same failure mode recorded for
  `matgl.load_model`, and it is exactly the DGL-compatibility path this package advertises.
  Missing keys now raise `ValueError` naming them. Unexpected keys only warn: they are the
  benign half of that compatibility (a checkpoint carrying extra heads this build lacks) and
  cannot leave a parameter uninitialised.

## ASE layer, element refs and missing labels (2026-09-29, fourth pass)

The rest of the review's findings. None is on the training path, so none affects a trained
model; most are on the relaxation/MD path, which the numerical checks never exercise.
`tests/test_ase_and_ref_regressions.py` covers all of them, and each test was confirmed to
fail with its own fix reverted.

* **Mixed periodicity silently discarded the cell.** `Atoms2Graph.get_graph` branched on
  `atoms.pbc.all()`, so a slab (`pbc=[True, True, False]`) took the isolated-molecule path:
  identity lattice, zero image offsets, Cartesian coordinates written into the `frac_coords`
  slot, and only in-cell neighbours. A 3-layer Si slab went from 168 edges to **18** — 89% of
  the neighbourhood gone, no warning — and the stress was divided by a unit volume instead of
  the real 120.08 Å³, i.e. wrong by ~120×. `find_points_in_spheres` accepts a per-axis `pbc`
  array, so mixed periodicity needs no special case: it now searches images only along the
  periodic axes (136 edges for that slab, between the fully-periodic 168 and the isolated 18)
  and keeps the real lattice. A periodic flag with a singular cell now raises instead of
  silently producing an undefined volume. Inherited from upstream matgl, not a port regression.

* **`ensemble="npt"` and `"npt_berendsen"` crashed on their own defaults.** Both barostats
  multiply by `compressibility_au`, which defaulted to `None`, so every run raised
  `TypeError: unsupported operand type(s) for *: 'float' and 'NoneType'` with no extra
  arguments at all. It is now derived from `bulk_modulus` (compressibility = 1/B) when not
  given, so these ensembles run out of the box and stay consistent with the barostat stiffness
  used by `npt_nose_hoover`.

* **`pfactor` was dimensionally wrong — the barostat was ~17× too stiff.** ASE defines
  `pfactor = ptime**2 * B`, with ptime in ASE time units and B a bulk modulus in eV/Å³. The
  code passed `75.0**2 * units.fs`: the time factor unsquared *and* the bulk modulus missing
  entirely. That gave **552.53** where "ptime = 75 fs, B = 0.6 eV/Å³" is **32.56** — a factor
  of 16.97, so the effective barostat timescale was ~309 fs rather than the documented 75 fs.
  The adjacent `ttime * units.fs` was correct all along, which is what made the `pfactor` line
  look right by association. There are now explicit `ptime` (fs) and `bulk_modulus` (eV/Å³)
  arguments; `pfactor` may still be passed directly to override both. Also present verbatim in
  upstream matgl `_ase_dgl.py`, so any published NPT comparison run through stock matgl shares
  the old behaviour.

* **`upper_triangular_cell()` rotated positions but not momenta.** `set_cell(scale_atoms=True)`
  carries positions into the new basis and leaves `momenta` in the old one, so velocities ended
  up misoriented relative to the lattice. Harmless for an isotropic Maxwell-Boltzmann seed —
  any orientation is as good as another — but wrong when continuing a trajectory or using an
  anisotropically prepared velocity field. The same linear map is now applied to the momenta;
  it is an orthogonal rotation, so kinetic energy is preserved exactly (verified).

* **`set_atoms` left the new Atoms at rest and the step counter stale.** It reassigned
  `dyn.atoms` and nothing else, so the incoming Atoms kept whatever velocities it had (usually
  none); the measured temperature was zero and the next `run()` died inside the thermostat with
  `ZeroDivisionError` while scaling by `T/T_old`. `dyn.nsteps` also carried over, so trajectory
  frames and log lines continued numbering from the previous run. Velocities are now reseeded
  at the configured temperature (only when the caller supplied none) and `nsteps` is reset.

* **`TrajectoryObserver` broke whenever stress or the cell was absent.** `__call__` appended to
  `stresses` only under `compute_stress` and to `cells` only for a periodic system, while
  `__getitem__` and `as_pandas` index all five lists by the same frame number. With
  `calc_stresses=False` a relaxation left `len(energies) == 2` against `len(stresses) == 0`, so
  `obs[0]` raised `IndexError` and `as_pandas()` raised `ValueError: All arrays must be of the
  same length`. All five lists now gain exactly one entry per frame, `None` where the quantity
  does not exist.

* **The final frame was recorded twice.** The observer attached with `interval=1` already fires
  on the last optimiser step, so the unconditional `obs()` after `run()` appended it again —
  `energies[-1] == energies[-2]` with identical positions — and any per-frame statistic over
  the trajectory (mean energy, step count, MSD) double-weighted the final configuration. It is
  now called only when the last step did not land on a recorded frame, which keeps it doing its
  job for `interval > 1`.

* **`AtomRef` with a 2D `property_offset` corrupted energies silently.** It ended in
  `torch.stack(offsets)[state_attr]`; `stack` is `(n_states, B)`, so indexing its *first* axis
  gathers along the state axis and returns a `(B, B)` matrix whose diagonal happens to hold the
  right answer. Worse, `Potential.forward` calls it **without** `state_attr`, so the index was
  `None` — which *adds* an axis rather than raising — giving `(1, n_states, B)`, and
  `total_energies + torch.squeeze(...)` then broadcast the energy itself to `(B, B)` with no
  error anywhere. It now gathers explicitly along the state axis, validates the indices, and
  raises when `state_attr` is missing. Latent in practice: `fit` only ever produces a 1D table,
  and that path is unchanged.

* **`allow_missing_labels=True` crashed whenever stress was disabled.** The NaN-mask loop
  covered every target, including stress, *before* the `calc_stresses` gate. With stress off
  the label is a full-length `zeros(B)` while `Potential.forward` returns a `torch.zeros(1)`
  placeholder, so masking raised `IndexError: The shape of the mask [B] ... does not match the
  shape of the indexed tensor [1]` — making the flag unusable in exactly the energy-and-forces
  configuration it is most wanted for. Masking now applies only to targets whose prediction is
  really shaped like its label; the per-target gates skip the rest anyway. No past result is
  affected, since `train.py` never sets the flag.

## Metric weighting, metric state and the bond norm (2026-09-29, third pass)

Three more defects from the same review. None changes what the model predicts; the first two
change what gets **reported**, which is arguably worse, because a wrong number that never
raises is the kind that ends up in a paper. `tests/test_metric_weighting_regressions.py`
covers all three; each test was confirmed to fail with its own fix reverted.

* **Force and stress metrics were weighted by the structure count.** `step` returned
  `preds[0].numel()` — the number of *structures* — and the mixin passed it as `batch_size`
  for every key in one `log_dict` call. Lightning uses `batch_size` as the weight when
  reducing an `on_epoch=True` metric, but `Force_MAE` is a mean over **atoms** and
  `Stress_MAE` over 3×3 tensors, so their epoch averages were weighted by the wrong quantity.
  A batch of 1 structure / 100 atoms at force MAE 1.0, reduced against 10 structures /
  10 atoms at 0.0, logged **0.0909** where the atom-weighted value is **0.9091** — a 10×
  misreport. Only heterogeneous batches show it, which is why uniform-cell runs never did.
  `loss_fn` now returns a per-metric weight map alongside the results, and `_log_results`
  groups the keys by weight and issues one `log_dict` per group. Energy keys and `Total_Loss`
  keep the structure count (`Total_Loss` is a weighted sum of incommensurate terms and has no
  natural element count). Weights are counted from the *valid* tensors, so
  `allow_missing_labels` masking is reflected.

* **The torchmetrics instances were never reset** (`grep -c '.reset()'` → 0). Called as
  functions they *return* the batch value — which is what Lightning logs, and is correct —
  while also folding the batch into internal state that was never cleared, so `compute()`
  returned a number pooled over every batch since construction (0.1 then 0.5 gives 0.3, and
  keeps drifting) and the state grew without bound. Nothing in the module calls `compute()`,
  which is exactly why the logged scalars looked fine and hid it. Now reset at every epoch
  boundary and for each stage, since the same instances are shared by train, validation and
  test. The scan is over `self.modules()`, **not** `vars(self)`: assigning an `nn.Module`
  attribute stores it in `_modules` rather than `__dict__`, so a `vars()` scan finds zero
  metrics and the reset is a silent no-op — a fix that looks right and does nothing.

* **The bond vector's own normalisation still used `torch.norm` + `clamp_min`** — the one
  place in `_diep_core.py` left using the pattern the rest of the file documents as
  forbidden, two lines above the comment block explaining why. The three candidate
  *perpendiculars* were carefully softened; the bond vector was not. A coincident or
  near-coincident bond (a duplicate site, or a self-image pair slipping past the converter's
  `bond_dist > numerical_tol` filter) gave a 1e12-magnitude first derivative and a **NaN
  Hessian row**, while its energy and force stayed finite and raised no warning. Now softened
  inside the sqrt like every other norm here. Unlike the other softened sites this one is not
  read through a `torch.where`, so the NaN stayed local to the offending edge rather than
  spreading to every frame. Forces, stresses and the Hessian on ordinary geometry are
  unchanged to float64 precision (verified: force and stress finite-difference agreement
  identical to pre-fix at 8.2e-12 and 3.3e-11; Hessian symmetry 1.0e-20).

## Triplet ordering (2026-09-29, third pass)

`layers/_diep_core.py` canonicalises each triplet so that the same physical triplet yields
the same descriptor however it was enumerated. The criterion was **purely geometric** —
longest edge by `argmax`, then the longer opposite leg — and neither step had a fallback for
a draw:

* `lengths.argmax` returns the **lowest index** on a tie, so an equilateral triplet chose its
  longest edge by enumeration position.
* `swap_mask = len_v_w + eps < len_u_w` never fires on equal legs (the `+ eps` is on the
  left), so an isoceles triplet kept whichever endpoint the caller enumerated first.

The module docstring claimed atoms were ordered by atomic number, but `atomic_numbers` was
only ever *read out* by the final `gather`, after the order was already fixed — it never got
a vote. Canonical coordinates matched between enumerations; the atomic numbers attached to
those coordinate slots did not, and `ordered_numbers` is what weights the Gaussian density.
One physical triplet therefore got two different descriptors:

```
enumeration A:  coords [(-1.3066,-0.3078), (1.3066,-0.3078), (0,0.6156)]  Z = [8, 7, 14]
enumeration B:  coords [(-1.3066,-0.3078), (1.3066,-0.3078), (0,0.6156)]  Z = [7, 8, 14]
```

This is not a corner case. `_triples_from_bonds` emits both `(b_i, b_k)` and `(b_k, b_i)` for
every centre atom, so **both enumerations of every triplet are always present in the same
batch**. Measured through the real integrator on a mixed O/Si/N cell, the two orderings of one
physical O–Si–N triple differed by **0.059 against a feature scale of 2.66 (~2%)**. Elemental
systems are unaffected (equal Z), which is why diamond-Si checks never caught it.

The fix consults atomic number **only to break a draw**, highest Z first, at both levels:

* the edge selection now ranks candidates lexicographically by (quantised length, higher-Z
  endpoint, lower-Z endpoint) via `_lexicographic_argmax`, instead of `argmax` alone;
* the endpoint ordering falls back to `z_v > z_u` when the legs are tied to within `eps`.

Equal Z needs no third level: the two atoms are then interchangeable, so either choice gives
the same (coordinate, charge) pairing and hence the same descriptor. Lengths are quantised by
`round(length / eps)` before ranking so that two edges within `eps` compare equal and the Z
keys decide — without it a 1e-12 Å difference would outrank a whole element, making the frame
hypersensitive to float noise exactly where it is meant to be stable.

Verified: the two enumerations of that O–Si–N triple now give **bit-identical** features
(0.0, was 0.059); 2000/2000 random untied triplets keep their previous ordering, so generic
geometry is untouched; forces and stresses still match central finite differences to ~1e-11;
rotation/translation invariance is now exact (0.0, improved from 3.6e-07, a side effect of
ranking on quantised keys rather than raw float norms). All selection happens under
`torch.no_grad()` — it feeds only comparisons and `argmax`, which are non-differentiable
anyway — while `pos_u`/`pos_v`/`pos_w` are re-gathered outside it, so gradients are unchanged.

`tests/test_canonicalisation_tiebreak.py` covers this; four of its seven tests were confirmed
to fail with the tie-break reverted (the other three are guards that must pass either way:
untied-ordering-unchanged, same-Z consistency, rotation invariance).

**This changes descriptors for tied triplets, so checkpoints trained before 2026-09-29 do not
mean the same thing on mixed-species data and should be retrained.**

## Bond-anchored triplet frame (2026-09-30)

The tie-break above makes the two enumerations of a triplet agree *at* a draw. It cannot make
the two *sides* of a draw agree. As two edge lengths cross, the canonical order flips, and
when the tied atoms are different elements the triangle is redrawn mirrored with its charges
swapped. The energy steps: 1.34 meV on an O–Ir–W triplet as O–W crosses O–Ir at 2.0 Å
(fold-0 epoch-88 checkpoint, float64). Autograd forces barely register the step, but it
breaks MD energy conservation and finite-difference checks. Any rule that picks one ordering
from lengths has this property.

`DIEP(triplet_frame="bond")` (in `train.py`, `--triplet-frame bond`) stops choosing.
`ThreeBodyInteractions` already sends each line-graph entry (first bond j→i, second bond j→k)
to its first bond. The frame draws the entry the way `DIEPIntegrator` draws that bond:
j on the left and i on the right along x, then k at
(r_ji·r_jk / |r_ji|, |r_ji × r_jk| / |r_ji|). The picture is then centred on its centroid,
as before.

Both enumerations of every triplet are in the line graph, so each bond still sees every
neighbour, from its own side. The two copies of a triplet therefore get different
descriptors, by design.

k always lands above the bond, but which side counts as "above" is arbitrary. So each grid
map is averaged with its mirror image in y. Without that fold the energy follows |y_k|, which
makes a cone at every 180° triplet, and the force flips direction there.

Measured (`tests/test_bond_anchored_triplets.py`, where each canonical case is the control
that shows the geometry really hits the defect):

| | canonical | bond |
|---|---|---|
| triplet-feature step across the O–Ir / O–W crossing, max ÷ median | 730 | 1.0 |
| finite-difference vs autograd force at the crossing, relative (h = 1e-4) | 0.28, growing as 1/h | 2e-9 |
| \|E − E₀\| / h through 180°, h = 1e-2 → 1e-3 | 1.39 → 1.36 (cone) | 0.038 → 0.0038 (smooth) |
| float32 vs float64 gradient, 179.9998° off-centre triplet | 2.7e3 (scale 8) | 4.8e-6 |

The frame uses only dot and cross products of the two bond vectors and never divides by a
vanishing length. So it also removes the canonical frame's float32 precision loss: there,
`perp / |perp|` on nearly straight, off-centre triplets reached 1.2% of MatPES structures,
with spurious forces up to 65 eV/Å. Parameter shapes are unchanged, and so is the cost: the
same training-step time on a 32-structure MatPES batch.

The two frames give different descriptors, so a model must be trained and evaluated in the
same one. `"canonical"` stays the default, so every existing checkpoint loads unchanged.
`"bond"` adds a `diep_integrator.bond_anchored_triplets` marker to the state dict, so a strict
load across frames fails instead of silently computing the wrong features. `DIEP.save` and
`DIEP.load` carry the setting in the model args. In `slurm/train.sh`,
`TRIPLET_FRAME=bond` writes to `runs_diep/diep_fold{k}_bond`.

## Numerical notes

### Vanishing norms and the Hessian

Norms that can vanish are softened *inside* the square root
(`sqrt(x.pow(2).sum() + eps**2)`) rather than by clamping the norm afterwards. `torch.norm`
at exactly zero has a finite first derivative but a **NaN second derivative**, and
`clamp_min` runs too late to prevent it. Because `torch.where` evaluates both branches and
propagates the gradient of the unselected one, a single bond or triplet lying along a
reference axis — routine in any axis-aligned cell — would otherwise make the *entire*
Hessian NaN, for every bond and triplet, while leaving energies and forces finite and
therefore giving no warning.

Two sites in `layers/_diep_core.py` need this, and both are fixed:

* `_canonicalize_triplets_batch` — the triplet frame (`bond_len`, `perp_norm`, and the
  `cross1` / `cross2` fallback axes).
* `_build_bond_frames_vectorized` — the **bond** frame's three candidate perpendiculars.
  `max_lengths` reads all three, so one zero-length trial poisons every edge.

Branch selectors are tested against the true (unsoftened) squared norms so the softening
never changes which branch is taken; in the bond frame those selectors are additionally
computed under `torch.no_grad()`, since `sqrt` of an exactly-zero squared length is NaN in
the *first* derivative and the comparisons need no gradient at all.

Energies are bit-identical to the unsoftened code; forces and stresses move by ~1e-10
(float32 rounding).

### Validating a Hessian

For a structure whose forces are smooth, the analytic Hessian agrees with central finite
differences of the forces to ~1e-14 in float64 (verified on diamond Si). Do **not** treat a
finite-difference mismatch as a Hessian bug without first checking that the force field is
actually differentiable at that geometry: the triplet canonicalisation orders atoms by length
and then by atomic number, and an atomic number is discrete, so as two lengths cross the
Z-to-slot assignment switches and the force field is genuinely discontinuous. The canonical
*coordinates* move continuously through the crossing (the jump scales with the perturbation);
it is `ordered_numbers` that flips. At such a point the finite-difference estimate *diverges*
as `h` shrinks (growing like 1/h) while the analytic Hessian stays perfectly symmetric.
Symmetry of the analytic Hessian is the more reliable check — it is never enforced, so
agreement to ~1e-19 is meaningful evidence. This discontinuity is a property of the
canonicalisation, not of the softening above, and is unchanged by it.

The crossing sits at the tie-break's `eps` (1e-6 Å) rather than exactly at the geometric tie.
It is unavoidable in the canonical frame: any rule assigning discrete labels to geometric slots
has to switch somewhere. `triplet_frame="bond"` assigns none and has no switch; see
*Bond-anchored triplet frame* above. The alternative — letting enumeration order decide, as the code did
before 2026-09-29 — is strictly worse, because it makes the descriptor depend on something
that is not physics at all. See *Triplet ordering* below.
