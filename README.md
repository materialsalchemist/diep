<img src="./assets/logo.svg" width="200px">

# Direct integration of the external potential, `diep`

`diep` is a material representation library that implements the ``direct integration of the external potential'' embedding for graph neural networks, as described here:

- [Sherif Abdulkader Tawfik, Tri Minh Nguyen, Salvy P. Russo, Truyen Tran, Sunil Gupta ORCID and Svetha Venkatesh, Embedding material graphs using the electron-ion potential: application to material fracture, Digital Discovery, 2024.](https://pubs.rsc.org/en/content/articlelanding/2024/dd/d4dd00246f)


## Installation

The base install uses the PyTorch Geometric (PyG) backend:

`pip install diep`

The original DGL backend is available as an optional extra (requires a platform DGL
and m3gnet support, e.g. not aarch64):

`pip install diep[dgl]`

# Features

`diep` is currently under active development.

## Two-body cutoff smoothing (PyG)

The PyG `DIEP` model applies the existing `polynomial_cutoff` to each bond's
raw DIEP feature vector before storing it in `g.rbf`. Both the edge embedding
and subsequent M3GNet graph-convolution layers consume these smoothed features:

```python
pair_cutoff = polynomial_cutoff(g.bond_dist, self.cutoff)
g.rbf = bond_features * pair_cutoff.unsqueeze(-1)
```

For `x = r / cutoff`, the default envelope is `1 - 10*x**3 + 15*x**4 - 6*x**5`
inside the cutoff and zero outside. Its value and first two derivatives vanish
at the cutoff. Raw DIEP bond features do not generally vanish there; multiplying
by this envelope makes them approach zero before graph construction removes a
bond. The factor acts over the full cutoff interval and applies to every channel
of a bond, in both `grid` and `sum` modes. It is enabled unconditionally in the
PyG model and does not depend on a `use_smooth` keyword.

No learnable parameters or feature dimensions are added. Three-body interactions
retain their existing product of cutoff factors for the two participating bonds.
The implementation is in `src/diep/pyg/models/_diep.py`; the DGL backend is
unchanged by this update. Consequently, backend parity tests explicitly apply
the same pair envelope to their DGL reference within the test fixture; they do
not assert that the two unmodified backend defaults still produce equal values.

Existing compatible weights can still load, but predictions and coordinate
gradients change. Re-evaluate predictive accuracy and consider fine-tuning or
retraining before using an existing model with the new smoothing in production
simulations.

### Validation

Local checks using the bundled `pretrained_models/diep_pes` weights verified
strict loading into PyG, finite energies/forces/stresses, individual-versus-batch
consistency, and forces against finite-difference energy gradients. These weights
use a 169-channel grid and include the saved energy scaling and elemental references.

In a controlled Li-O pair test with the same weights and a graph rebuilt at every
distance, `|E(5-h) - E(5+h)|` was approximately `0.11015 eV` without pair smoothing
at `h = 1e-4 Å`, versus `9.77e-15 eV` with smoothing (float64). The corresponding
smoothed basis values and their first two distance derivatives vanished at the
cutoff within numerical precision. This is a cutoff-behavior check, not a test-set
accuracy benchmark or MD validation; it does not address the separate triplet
projection issue involving second coordinate derivatives.

Focused regression tests cover both integration modes, the cutoff endpoint,
and energy/force behavior when a pair crosses the graph cutoff:

```bash
PYTHONPATH=src python -m pytest --confcutdir=tests/pyg tests/pyg/test_pair_cutoff.py -q
```
