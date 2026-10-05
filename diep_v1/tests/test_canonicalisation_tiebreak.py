"""Regression tests for the triplet canonicalisation tie-break, added 2026-09-29.

``_canonicalize_triplets_batch`` orders a triplet's three atoms so that the same physical
triplet yields the same descriptor no matter how it was enumerated. Its criterion was purely
geometric -- longest edge by ``argmax``, then the longer opposite leg -- and neither step had
a fallback for a draw:

* ``lengths.argmax`` returns the *lowest index* on a tie, so an equilateral triplet chose its
  longest edge by enumeration position.
* ``swap_mask = len_v_w + eps < len_u_w`` never fires on equal legs, so an isoceles triplet
  kept whichever endpoint the caller happened to enumerate first.

The module docstring claimed atoms were ordered by atomic number, but ``atomic_numbers`` was
only ever *read out* by the final ``gather``, after the order was already fixed -- it never
got a vote. Canonical *coordinates* matched between enumerations; the atomic numbers attached
to those coordinate slots did not, and ``ordered_numbers`` is what weights the Gaussian
density in ``DIEPIntegrator``. One physical triplet therefore got two descriptors.

This matters because ``_triples_from_bonds`` emits both ``(b_i, b_k)`` and ``(b_k, b_i)`` for
every centre atom, so both enumerations of every triplet are always present in the same
batch. Measured on a mixed O/Si/N cell before the fix: the two orderings of one physical
O-Si-N triple differed by 0.059 against a feature scale of 2.66.

Each test below was confirmed to fail with the tie-break reverted.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from diep_pyg import config
from diep_pyg.graph.compute import compute_pair_vector_and_distance, create_line_graph
from diep_pyg.graph.converters import Structure2Graph
from diep_pyg.layers._diep import DIEPIntegrator
from diep_pyg.layers._diep_core import _canonicalize_triplets_batch

# An isoceles triplet: the two legs are exactly equal, so the leg comparison is a draw and
# only the atomic numbers can break it. 109.5 degrees and 1.6 A are ordinary bond geometry.
_HALF_ANGLE = math.radians(109.5 / 2)
_LEG = 1.60
_NBR_A = [-_LEG * math.sin(_HALF_ANGLE), -_LEG * math.cos(_HALF_ANGLE), 0.0]
_NBR_B = [_LEG * math.sin(_HALF_ANGLE), -_LEG * math.cos(_HALF_ANGLE), 0.0]
_CENTRE = [0.0, 0.0, 0.0]


def test_isoceles_triplet_is_invariant_to_enumeration_order():
    """Equal legs: the two enumerations of one triplet must agree on the (coord, Z) pairing.

    Both orderings are always emitted together by ``_triples_from_bonds``, so a draw settled
    by enumeration order gives one physical triplet two different descriptors.
    """
    coords_a = torch.tensor([[_NBR_A, _CENTRE, _NBR_B]], dtype=torch.float64)
    coords_b = torch.tensor([[_NBR_B, _CENTRE, _NBR_A]], dtype=torch.float64)
    # O at the first-listed neighbour in A, N in B -- the same physical O-Si-N triplet.
    numbers_a = torch.tensor([[8.0, 14.0, 7.0]], dtype=torch.float64)
    numbers_b = torch.tensor([[7.0, 14.0, 8.0]], dtype=torch.float64)

    canon_a, ordered_a = _canonicalize_triplets_batch(coords_a, numbers_a)
    canon_b, ordered_b = _canonicalize_triplets_batch(coords_b, numbers_b)

    # The coordinates always matched; it was the atomic numbers that swapped.
    assert torch.allclose(canon_a, canon_b, atol=1e-12)
    assert torch.equal(ordered_a, ordered_b), (
        f"same triplet, different Z ordering: {ordered_a.tolist()} vs {ordered_b.tolist()}"
    )
    # Highest Z first among the tied pair, so the higher-Z neighbour takes the earlier slot.
    assert ordered_a[0, 0] >= ordered_a[0, 1]


def test_equilateral_triplet_is_invariant_to_enumeration_order():
    """All three edges tied: ``argmax`` alone picked the longest edge by position.

    The isoceles case only exercises the leg comparison. An equilateral triplet of three
    distinct species also ties the *edge* selection, which is a separate draw one level up.
    """
    side = 2.0
    p0 = [0.0, 0.0, 0.0]
    p1 = [side, 0.0, 0.0]
    p2 = [side / 2, side * math.sqrt(3) / 2, 0.0]

    # Three cyclic relabellings of one equilateral O/Si/N triangle.
    enumerations = [
        ([p0, p1, p2], [8.0, 14.0, 7.0]),
        ([p1, p2, p0], [14.0, 7.0, 8.0]),
        ([p2, p0, p1], [7.0, 8.0, 14.0]),
    ]
    results = [
        _canonicalize_triplets_batch(
            torch.tensor([coords], dtype=torch.float64),
            torch.tensor([numbers], dtype=torch.float64),
        )
        for coords, numbers in enumerations
    ]

    first_coords, first_numbers = results[0]
    for coords, numbers in results[1:]:
        assert torch.allclose(coords, first_coords, atol=1e-12)
        assert torch.equal(numbers, first_numbers), (
            f"equilateral triplet ordered by enumeration position: "
            f"{numbers.tolist()} vs {first_numbers.tolist()}"
        )


def test_same_z_tie_is_consistent():
    """Equal legs *and* equal Z: the two atoms are interchangeable, so either choice is fine.

    There is deliberately no third tie-break level. What must still hold is that the result
    does not depend on enumeration order -- which it cannot, since swapping two identical
    atoms is not an observable change.
    """
    coords_a = torch.tensor([[_NBR_A, _CENTRE, _NBR_B]], dtype=torch.float64)
    coords_b = torch.tensor([[_NBR_B, _CENTRE, _NBR_A]], dtype=torch.float64)
    numbers = torch.tensor([[8.0, 14.0, 8.0]], dtype=torch.float64)  # O-Si-O

    canon_a, ordered_a = _canonicalize_triplets_batch(coords_a, numbers)
    canon_b, ordered_b = _canonicalize_triplets_batch(coords_b, numbers)
    assert torch.allclose(canon_a, canon_b, atol=1e-12)
    assert torch.equal(ordered_a, ordered_b)


def test_untied_triplets_keep_their_previous_ordering():
    """The tie-break must engage *only* on a draw, leaving generic geometry untouched.

    Reimplements the pre-fix ordering and requires agreement on random triplets, where a
    genuine draw has probability zero. A tie-break that also reordered untied triplets would
    silently change every descriptor in the model.
    """
    torch.manual_seed(0)
    n = 512
    coords = torch.randn(n, 3, 3, dtype=torch.float64) * 2.0
    numbers = torch.randint(1, 30, (n, 3)).to(torch.float64)
    eps = 1e-6

    # Pre-fix ordering: geometric only, ties falling through to enumeration order.
    lengths = torch.stack(
        [
            (coords[:, 0] - coords[:, 1]).norm(dim=1),
            (coords[:, 1] - coords[:, 2]).norm(dim=1),
            (coords[:, 0] - coords[:, 2]).norm(dim=1),
        ],
        dim=1,
    )
    pair_indices = torch.tensor([[0, 1], [1, 2], [0, 2]])
    longest = pair_indices[lengths.argmax(dim=1)]
    u_idx, v_idx = longest[:, 0], longest[:, 1]
    w_idx = 3 - u_idx - v_idx
    rows = torch.arange(n)
    len_u_w = (coords[rows, u_idx] - coords[rows, w_idx]).norm(dim=1)
    len_v_w = (coords[rows, v_idx] - coords[rows, w_idx]).norm(dim=1)
    swap = len_v_w + eps < len_u_w
    u_old, v_old = torch.where(swap, v_idx, u_idx), torch.where(swap, u_idx, v_idx)
    expected = torch.gather(numbers, 1, torch.stack([u_old, v_old, 3 - u_old - v_old], dim=1))

    _, ordered = _canonicalize_triplets_batch(coords, numbers)
    assert torch.equal(ordered, expected)


def test_tie_break_is_rotation_invariant():
    """The tie-break keys on Z and on quantised lengths, both frame-independent."""
    from scipy.spatial.transform import Rotation

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    coords = torch.randn(64, 3, 3, dtype=torch.float64) * 2.0
    numbers = torch.randint(1, 30, (64, 3)).to(torch.float64)
    reference_coords, reference_numbers = _canonicalize_triplets_batch(coords, numbers)

    for _ in range(8):
        rotation = torch.tensor(Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix())
        shifted = coords @ rotation.T + torch.tensor(rng.normal(size=3))
        rotated_coords, rotated_numbers = _canonicalize_triplets_batch(shifted, numbers)
        assert torch.equal(rotated_numbers, reference_numbers)
        assert torch.allclose(rotated_coords, reference_coords, atol=1e-9)


def test_batched_mix_of_tied_and_untied_triplets():
    """A tied triplet must be settled correctly while sharing a batch with untied ones.

    The selection is vectorised over the batch, so a per-row tie-break is the kind of thing
    that works in isolation and breaks once rows are mixed.
    """
    torch.manual_seed(0)
    generic = (torch.randn(3, 3, 3, dtype=torch.float64) * 2.0).tolist()
    coords = torch.tensor(
        [[_NBR_A, _CENTRE, _NBR_B], *generic, [_NBR_B, _CENTRE, _NBR_A]], dtype=torch.float64
    )
    numbers = torch.tensor(
        [[8.0, 14.0, 7.0], *torch.randint(1, 30, (3, 3)).to(torch.float64).tolist(), [7.0, 14.0, 8.0]],
        dtype=torch.float64,
    )
    _, ordered = _canonicalize_triplets_batch(coords, numbers)
    # First and last rows are the two enumerations of the same physical triplet.
    assert torch.equal(ordered[0], ordered[-1])


def test_both_orderings_of_one_triple_give_identical_features():
    """End to end: the defect's observable signature, through the real integrator.

    ``_triples_from_bonds`` emits both orderings of every triple, so the integrator sees them
    in the same batch. Before the tie-break these differed by 0.059 on a feature scale of
    2.66; they must now be bit-identical.
    """
    float_th = config.float_th
    config.float_th = torch.float64
    try:
        half_angle, leg, box = math.radians(109.5 / 2), 1.7, 8.0
        nbr_a = np.array([-leg * math.sin(half_angle), -leg * math.cos(half_angle), 0.0]) / box
        nbr_b = np.array([leg * math.sin(half_angle), -leg * math.cos(half_angle), 0.0]) / box
        elements = ("O", "Si", "N")
        structure = Structure(Lattice.cubic(box), list(elements), [nbr_a, [0, 0, 0], nbr_b])

        graph, lattice, _ = Structure2Graph(element_types=elements, cutoff=5.0).get_graph(structure)
        lattice = torch.as_tensor(lattice[0], dtype=torch.float64)
        graph.pbc_offshift = graph.pbc_offset @ lattice
        graph.pos = graph.frac_coords @ lattice
        graph.bond_vec, graph.bond_dist = compute_pair_vector_and_distance(graph)
        create_line_graph(graph, 4.5)

        atomic_numbers = torch.tensor([8.0, 14.0, 7.0], dtype=torch.float64)[graph.node_type.long()]
        _, triplet_features = DIEPIntegrator(mode="grid").double()(
            graph, atomic_numbers, compute_triplets=True
        )

        src, dst = graph.edge_index
        first, second = graph.triple_index
        # The two rows describing the O-Si-N triple centred on Si, in opposite order.
        matching = [
            t
            for t in range(first.numel())
            if int(src[int(first[t])]) == 1
            and {int(dst[int(first[t])]), int(dst[int(second[t])])} == {0, 2}
        ]
        assert len(matching) == 2, f"expected both orderings, found {len(matching)}"
        difference = (triplet_features[matching[0]] - triplet_features[matching[1]]).abs().max()
        assert float(difference) == pytest.approx(0.0, abs=1e-12), (
            f"the two enumerations of one physical triple differ by {float(difference):g}"
        )
    finally:
        config.float_th = float_th
