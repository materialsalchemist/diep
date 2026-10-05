"""Framework-neutral geometry behind the DIEP integrator.

Grid construction, Gaussian densities, local bond frames and the two triplet frames
(canonicalised, or anchored on the receiving bond) -- all of it plain tensor code that never
touches a graph object. The graph-bound ``DIEPIntegrator``
that consumes these lives in :mod:`diep_pyg.layers._diep`.

Only the batched frame/canonicalisation routines live here. The per-bond and per-triplet
scalar twins (``_build_bond_frame``, ``_canonicalize_triplet`` and the ``LocalFrame2D`` /
``_normalize`` / ``_choose_perpendicular_unit`` / ``_project_points`` helpers they needed) were
removed: nothing called them, and they took plain ``torch.norm`` of quantities that vanish for
axis-aligned bonds and collinear triplets -- exactly the NaN-second-derivative trap the
batched versions below are carefully softened against. Keeping an unsoftened copy around
invited reintroducing that bug by reuse.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from diep_pyg import config


def _compute_delta_area(grid_axis: torch.Tensor) -> torch.Tensor:
    """Return the area element for a uniform Cartesian grid."""
    if grid_axis.numel() < 2:
        return torch.tensor(1.0, dtype=config.float_th, device=grid_axis.device)
    step = grid_axis[1] - grid_axis[0]
    return step * step


@dataclass
class DIEPGrid:
    """Utility container storing the 2D grid definition."""

    half_length: float
    spacing: float
    device: torch.device

    def __post_init__(self):
        num_points = int(round(2 * self.half_length / self.spacing)) + 1
        axis = torch.linspace(
            -self.half_length,
            self.half_length,
            steps=num_points,
            dtype=config.float_th,
            device=self.device,
        )
        self.axis = axis
        self.points = torch.stack(torch.meshgrid(axis, axis, indexing="xy"), dim=-1).reshape(-1, 2)
        self.delta_area = _compute_delta_area(axis)


def _gaussian_density(diff_sq: torch.Tensor, sigma: torch.Tensor | float) -> torch.Tensor:
    """Compute Gaussian electron density contribution with squared-distance decay.

    ``sigma`` may be a scalar or a tensor of per-channel widths; a tensor of shape ``(D,)``
    broadcasts against a trailing size-1 axis of ``diff_sq`` to produce ``D`` density channels
    from the same squared distances.

    Returns normalized density for 2D integration.
    """
    denom = sigma if torch.is_tensor(sigma) else torch.tensor(sigma, dtype=diff_sq.dtype, device=diff_sq.device)
    denom = denom.to(dtype=diff_sq.dtype, device=diff_sq.device)
    # Normalize by (pi * sigma) for proper 2D Gaussian normalization
    normalization = torch.pi * denom
    return torch.exp(-torch.clamp(diff_sq, min=0.0) / denom) / normalization


def _build_bond_frames_vectorized(pos_src: torch.Tensor, pos_dst: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct local 2D frames for all bonded pairs in a vectorized manner.
    
    Args:
        pos_src: (N_edges, 3) source atom positions
        pos_dst: (N_edges, 3) destination atom positions
    
    Returns:
        origins: (N_edges, 3) frame origins (midpoints)
        e_x_batch: (N_edges, 3) x-axis unit vectors
        e_y_batch: (N_edges, 3) y-axis unit vectors
    """
    # Compute bond vectors and origins
    vecs = pos_dst - pos_src  # (N_edges, 3)
    origins = 0.5 * (pos_src + pos_dst)  # (N_edges, 3)
    
    # Normalize bond vectors to get e_x. Softened *inside* the square root, like every other
    # norm in this file, rather than by clamping afterwards: `torch.norm` at exactly zero has
    # a finite first derivative but a NaN second one, and a `clamp_min` on the result runs too
    # late to prevent that. A coincident (or near-coincident) bond -- a duplicate site, or a
    # self-image pair slipping past the converter's `bond_dist > numerical_tol` filter --
    # otherwise gave a 1e12-magnitude first derivative and a NaN Hessian row for that edge,
    # while its energy and force stayed finite and so raised no warning. Unlike the softened
    # sites below this one is not read through a `torch.where`, so the NaN stayed local to the
    # offending edge rather than spreading to every frame. The eps**2 floor shifts each length
    # by less than eps.
    eps_len = 1e-10
    norms = torch.sqrt((vecs**2).sum(dim=1, keepdim=True) + eps_len**2)  # (N_edges, 1)
    e_x_batch = vecs / norms  # (N_edges, 3)
    
    # Choose perpendicular vectors for e_y (vectorized)
    # Try [1,0,0] first, then [0,1,0], then [0,0,1]
    device = pos_src.device
    dtype = pos_src.dtype
    n_edges = pos_src.shape[0]
    
    candidate_1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).expand(n_edges, 3)
    candidate_2 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).expand(n_edges, 3)
    candidate_3 = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device).expand(n_edges, 3)
    
    # Gram-Schmidt: subtract projection onto e_x
    def make_perpendicular(candidate, e_x):
        proj = (candidate * e_x).sum(dim=1, keepdim=True)  # (N_edges, 1)
        perp = candidate - proj * e_x  # (N_edges, 3)
        return perp
    
    # Lengths are softened *inside* the square root rather than taken with `torch.norm`.
    # A bond lying along one of the candidate axes -- routine in any axis-aligned cell --
    # makes that trial vector exactly zero, and `torch.norm` at zero has a finite first
    # derivative but a NaN second derivative. `max_lengths` below reads all three lengths,
    # so that NaN reaches every edge's frame and makes the whole Hessian NaN, while
    # energies and forces stay finite and give no warning. The eps**2 floor changes each
    # length by less than eps and leaves the second derivative finite. Selection still uses
    # the true squared lengths, so which trial wins is exactly what it was before.
    eps = 1e-10
    trial_1 = make_perpendicular(candidate_1, e_x_batch)
    sq_1 = (trial_1**2).sum(dim=1, keepdim=True)
    lengths_1 = torch.sqrt(sq_1 + eps**2)

    trial_2 = make_perpendicular(candidate_2, e_x_batch)
    sq_2 = (trial_2**2).sum(dim=1, keepdim=True)
    lengths_2 = torch.sqrt(sq_2 + eps**2)

    trial_3 = make_perpendicular(candidate_3, e_x_batch)
    sq_3 = (trial_3**2).sum(dim=1, keepdim=True)
    lengths_3 = torch.sqrt(sq_3 + eps**2)

    # Use the trial with the largest length (most perpendicular). Selection is done on
    # detached true lengths: these feed only boolean comparisons, so they need no gradient,
    # and `torch.sqrt` of an exactly-zero squared length is NaN in the *first* derivative --
    # detaching keeps that out of the graph entirely.
    with torch.no_grad():
        true_1, true_2, true_3 = torch.sqrt(sq_1), torch.sqrt(sq_2), torch.sqrt(sq_3)
        max_lengths = torch.maximum(torch.maximum(true_1, true_2), true_3)

        # Select best trial for each edge. There is no explicit `use_3`: the nested
        # `torch.where` below falls through to trial 3 whenever neither 1 nor 2 is chosen,
        # which is the same set the old `~use_1 & ~use_2` mask selected.
        use_1 = (true_1 >= max_lengths - 1e-10).squeeze()
        use_2 = (true_2 >= max_lengths - 1e-10).squeeze() & ~use_1

    # Built with torch.where rather than masked assignment: an in-place write into a
    # zeros tensor would still have to divide by the unselected trials' lengths.
    e_y_batch = torch.where(
        use_1.unsqueeze(-1),
        trial_1 / lengths_1,
        torch.where(use_2.unsqueeze(-1), trial_2 / lengths_2, trial_3 / lengths_3),
    )

    return origins, e_x_batch, e_y_batch


def _project_points_batch(points: torch.Tensor, origins: torch.Tensor, e_x: torch.Tensor, e_y: torch.Tensor) -> torch.Tensor:
    """Project 3D coordinates into local 2D frames (vectorized).
    
    Args:
        points: (N_edges, N_atoms_per_edge, 3) 3D coordinates
        origins: (N_edges, 3) frame origins
        e_x: (N_edges, 3) x-axis vectors
        e_y: (N_edges, 3) y-axis vectors
    
    Returns:
        coords_2d: (N_edges, N_atoms_per_edge, 2) 2D coordinates
    """
    diff = points - origins.unsqueeze(1)  # (N_edges, N_atoms, 3)
    x = (diff * e_x.unsqueeze(1)).sum(dim=-1)  # (N_edges, N_atoms)
    y = (diff * e_y.unsqueeze(1)).sum(dim=-1)  # (N_edges, N_atoms)
    return torch.stack([x, y], dim=-1)  # (N_edges, N_atoms, 2)


def _lexicographic_argmax(keys: torch.Tensor) -> torch.Tensor:
    """Index of the lexicographically largest key row, per batch element.

    Args:
        keys: ``(n, k, d)`` tensor of ``k`` candidates each carrying a ``d``-component key,
            ordered most significant component first.

    Returns:
        ``(n,)`` index of the winning candidate. Ties across *every* component fall back to
        the lowest index, which is safe only because the callers arrange for fully tied
        candidates to be interchangeable.

    Comparing keys one component at a time rather than packing them into a single scalar:
    packing needs a bound on each component's range to pick multipliers, and a wrong bound
    silently lets a low-significance component outrank a high-significance one.
    """
    n, k, _ = keys.shape
    alive = torch.ones((n, k), dtype=torch.bool, device=keys.device)
    for component in range(keys.size(-1)):
        values = keys[..., component]
        masked = values.masked_fill(~alive, float("-inf"))
        best = masked.max(dim=1, keepdim=True).values
        # Keep every candidate still tied for the best value of this component, then let the
        # next component decide among them.
        alive = alive & (values >= best)
    return torch.argmax(alive.to(torch.uint8), dim=1)


def _canonicalize_triplets_batch(
    coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised canonicalisation for triplets following DIEP conventions.

    Atoms are ordered by geometry first, with atomic number consulted only to break a draw:

    1. the longest edge becomes ``(u, v)``; ties among edges go to the one whose endpoints
       carry the higher atomic numbers;
    2. within that edge, the endpoint opposite the longer leg comes first; on equal legs the
       higher-Z endpoint comes first;
    3. equal Z needs no further rule -- the two atoms are then interchangeable, so either
       choice produces the same (coordinate, charge) pairing and the same descriptor.

    Both tie-breaks matter because ``_triples_from_bonds`` emits both orderings of every
    triplet into the same batch, so a draw settled by enumeration order would give one
    physical triplet two different descriptors. See the README's *Triplet ordering* section.

    Args:
        coords: Tensor of shape (n_triplets, 3, 3) containing Cartesian coordinates of the three atoms
            forming each triplet. The second axis is ordered as (neighbor_i, center, neighbor_k).
        atomic_numbers: Tensor of shape (n_triplets, 3) with the atomic numbers in the same ordering as coords.
        eps: Numerical stability tolerance.

    Returns:
        canonical_coords: Tensor of shape (n_triplets, 3, 2) containing 2D coordinates in the canonical frame.
        ordered_numbers: Tensor of shape (n_triplets, 3) containing atomic numbers reordered to match
            canonical_coords.
    """
    if coords.numel() == 0:
        canonical = torch.zeros((0, 3, 2), dtype=config.float_th, device=coords.device)
        numbers = torch.zeros((0, 3), dtype=config.float_th, device=coords.device)
        return canonical, numbers

    device = coords.device
    dtype = coords.dtype
    n_triplets = coords.shape[0]

    # Identify the longest edge (u, v) for each triplet.
    #
    # Both selections below are ties waiting to happen, and a tie broken by anything other
    # than the atoms' own identities lets the caller's *enumeration order* decide the
    # canonical frame. `_triples_from_bonds` emits both (b_i, b_k) and (b_k, b_i) for every
    # centre atom, so both enumerations of a triplet are always present in the same batch:
    # a tie resolved by index gives one physical triplet two different descriptors.
    #
    # The geometric criterion stays primary. Atomic number is consulted only to break a
    # draw, highest Z first, because Z belongs to the atom rather than to the order it was
    # enumerated in and so resolves the draw identically every time. Equal Z needs no third
    # level: the two atoms are then interchangeable, and either choice yields the same
    # (coordinate, charge) pairing and hence the same descriptor.
    #
    # Selection is done under `no_grad` on true (unsoftened) norms: these feed only
    # comparisons and `argmax`, which are non-differentiable anyway, so keeping them out of
    # the graph costs nothing and avoids leaving a `norm`-at-zero in it for coincident atoms.
    vec01 = coords[:, 0] - coords[:, 1]
    vec12 = coords[:, 1] - coords[:, 2]
    vec02 = coords[:, 0] - coords[:, 2]
    batch_range = torch.arange(n_triplets, device=device)
    pair_indices = torch.tensor([[0, 1], [1, 2], [0, 2]], device=device)

    with torch.no_grad():
        lengths = torch.stack(
            [torch.norm(vec01, dim=1), torch.norm(vec12, dim=1), torch.norm(vec02, dim=1)],
            dim=1,
        )
        # Rank the three candidate edges by length, then by the Z multiset of their two
        # endpoints. `argmax` alone returns the lowest index on a tie, which is exactly the
        # enumeration-order leak: an equilateral triplet of three distinct species has all
        # three edges tied and picked its longest edge by position.
        # (n, 3, 2): for each triplet, the Z of the two endpoints of each candidate edge.
        z_pairs = atomic_numbers[batch_range[:, None, None], pair_indices[None, :, :]]
        z_hi = z_pairs.max(dim=2).values.to(lengths.dtype)  # higher-Z endpoint of each edge
        z_lo = z_pairs.min(dim=2).values.to(lengths.dtype)  # lower-Z endpoint of each edge
        # Quantise the length so two edges within `eps` compare equal and the Z keys decide.
        # Without this a 1e-12 A difference would outrank a whole element, making the frame
        # hypersensitive to float noise exactly where it is meant to be stable.
        length_key = torch.round(lengths / eps)
        edge_rank = torch.stack([length_key, z_hi, z_lo], dim=-1)
        best_edge = _lexicographic_argmax(edge_rank)
        longest_pair = pair_indices[best_edge]
        u_idx = longest_pair[:, 0]
        v_idx = longest_pair[:, 1]
        w_idx = 3 - u_idx - v_idx

        # Order the chosen edge's endpoints: longer opposite leg first, then higher Z first.
        # `len_v_w + eps < len_u_w` alone never fires on equal legs, so an isoceles triplet
        # kept whichever endpoint the caller happened to enumerate first.
        len_u_w = torch.norm(coords[batch_range, u_idx] - coords[batch_range, w_idx], dim=1)
        len_v_w = torch.norm(coords[batch_range, v_idx] - coords[batch_range, w_idx], dim=1)
        z_u = atomic_numbers[batch_range, u_idx].to(lengths.dtype)
        z_v = atomic_numbers[batch_range, v_idx].to(lengths.dtype)
        legs_tied = (len_v_w - len_u_w).abs() <= eps
        swap_mask = torch.where(legs_tied, z_v > z_u, len_v_w + eps < len_u_w)
        u_idx, v_idx = torch.where(swap_mask, v_idx, u_idx), torch.where(swap_mask, u_idx, v_idx)
        w_idx = 3 - u_idx - v_idx

    pos_u = coords[batch_range, u_idx]
    pos_v = coords[batch_range, v_idx]
    pos_w = coords[batch_range, w_idx]

    order = torch.stack([u_idx, v_idx, w_idx], dim=1)
    ordered_numbers = torch.gather(atomic_numbers, 1, order)

    origin = 0.5 * (pos_u + pos_v)
    vec_uv = pos_v - pos_u
    # Softened under the sqrt like the norms below. `vec_uv` spans the longest edge of the
    # triangle, so it only vanishes for exactly coincident atoms, but the softened form is
    # free and keeps the second derivative finite there too.
    bond_len = torch.sqrt((vec_uv**2).sum(dim=1, keepdim=True) + eps**2)
    e_x = vec_uv / bond_len

    vec_w = pos_w - origin
    proj = (vec_w * e_x).sum(dim=1, keepdim=True) * e_x
    perp = vec_w - proj
    # Softened inside the sqrt for the same reason as the fallback norms below: a collinear
    # triplet makes `perp` exactly zero, and a plain `torch.norm` there would hand back a
    # NaN second derivative that `torch.where` then spreads to every triplet. The eps**2
    # floor also makes the value strictly positive, so the `use_fallback` test below keeps
    # selecting the fallback in exactly the degenerate cases it did before.
    perp_sq = (perp**2).sum(dim=1, keepdim=True)
    perp_norm = torch.sqrt(perp_sq + eps**2)
    # Tested against the true norm, not the softened one, so the set of triplets routed to
    # the fallback is exactly what it was before the softening.
    use_fallback = perp_sq.squeeze(-1) < eps**2

    # Fallback axes for degenerate configurations.
    #
    # The norms below are softened *inside* the square root rather than by clamping the
    # norm afterwards. `torch.norm(x)` at exactly x = 0 has a finite first derivative but a
    # NaN second derivative, and `clamp_min` runs after the norm, so it never prevents that
    # NaN. Because `torch.where` evaluates both branches and propagates the gradient of the
    # unselected one, a single triplet whose bond lies along `ref1` (cross product exactly
    # zero -- routine in any axis-aligned cell) makes the whole Hessian NaN, for every
    # triplet, even those that never take the fallback. Adding eps**2 under the sqrt keeps
    # the value identical to within eps while leaving the second derivative finite.
    ref1 = torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device).expand_as(e_x)
    ref2 = torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device).expand_as(e_x)
    cross1 = torch.cross(e_x, ref1, dim=1)
    cross1_sq = (cross1**2).sum(dim=1, keepdim=True)
    cross1_norm = torch.sqrt(cross1_sq + eps**2)
    cross2 = torch.cross(e_x, ref2, dim=1)
    cross2_norm = torch.sqrt((cross2**2).sum(dim=1, keepdim=True) + eps**2)
    # Selected on the true squared norm: the softened `cross1_norm` is >= eps by
    # construction, so testing it against eps would always pick `cross1` and never fall
    # through to `cross2` for a bond lying along `ref1` -- the one case this branch exists
    # to handle.
    e_y_fallback = torch.where(
        cross1_sq > eps**2,
        cross1 / cross1_norm,
        cross2 / cross2_norm,
    )
    e_y = torch.where(
        use_fallback.unsqueeze(-1),
        e_y_fallback,
        perp / perp_norm,  # perp_norm >= eps by construction, so no clamp is needed
    )
    orientation = (vec_w * e_y).sum(dim=1, keepdim=True)
    flip_mask = orientation < 0
    e_y = torch.where(flip_mask, -e_y, e_y)

    diff = coords - origin.unsqueeze(1)
    x = (diff * e_x.unsqueeze(1)).sum(dim=-1)
    y = (diff * e_y.unsqueeze(1)).sum(dim=-1)
    projected = torch.stack([x, y], dim=-1)

    gather_idx = order.unsqueeze(-1).expand(-1, -1, 2)
    canonical = torch.gather(projected, 1, gather_idx)
    translation = canonical.mean(dim=1, keepdim=True)
    canonical = canonical - translation

    return canonical.to(config.float_th), ordered_numbers.to(config.float_th)


def _bond_anchored_triplets_batch(
    coords: torch.Tensor,
    atomic_numbers: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw each triplet in the frame of its first bond, choosing nothing from the geometry.

    The alternative to :func:`_canonicalize_triplets_batch`, selected by
    ``DIEPIntegrator(triplet_frame="bond")``. Canonicalisation picks, per triplet, which edge
    lies on the x-axis and which of its ends sits on the left, from lengths with atomic number
    as tie-break. Any such pick is discontinuous: when two lengths cross and the tied atoms
    differ in Z, the triangle is redrawn mirrored with the charges swapped, and the energy
    steps (1.34 meV on an O-Ir-W triplet whose two O legs cross at 2.0 A).

    Here nothing is picked. A line-graph entry (first bond j->i, second bond j->k) already says
    which bond receives the message -- ``ThreeBodyInteractions`` scatters onto ``first`` -- so
    the triplet is drawn the way ``DIEPIntegrator`` draws that bond, centre j on the left and i
    on the right along x, with k added at height ``|r_ji x r_jk| / r_ji``. Both orderings of
    every triplet are in the line graph, so each bond still sees every neighbour, from its own
    side. The two copies of a triplet therefore get different descriptors, by design.

    Coordinates come from dot and cross products of the two bond vectors rather than from a
    projected in-plane axis. Nothing divides by a vanishing length, so a near-collinear
    triplet keeps full float32 precision, which the canonical frame's ``perp / |perp|`` does
    not.

    k always lands at y >= 0. Which side of the bond counts as "up" is the one choice left,
    and the integrator removes it by averaging each feature map with its mirror image in y.

    Args:
        coords: (n_triplets, 3, 3) Cartesian coordinates ordered (neighbor_i, center,
            neighbor_k): the end of the first bond, the shared centre, the end of the second.
        atomic_numbers: (n_triplets, 3) atomic numbers in the same order.
        eps: softening length (A) that keeps the second derivative finite for coincident or
            exactly collinear atoms.

    Returns:
        coords_2d: (n_triplets, 3, 2) planar coordinates in the input order, centred on the
            triangle's centroid like the canonical frame.
        numbers: (n_triplets, 3) the atomic numbers, order unchanged.
    """
    if coords.numel() == 0:
        planar = torch.zeros((0, 3, 2), dtype=config.float_th, device=coords.device)
        numbers = torch.zeros((0, 3), dtype=config.float_th, device=coords.device)
        return planar, numbers

    first = coords[:, 0] - coords[:, 1]  # centre -> end of the receiving bond
    second = coords[:, 2] - coords[:, 1]  # centre -> end of the other bond
    length = torch.sqrt((first**2).sum(dim=1) + eps**2)
    x_k = (first * second).sum(dim=1) / length
    cross = torch.cross(first, second, dim=1)
    # Softened inside the sqrt for the reason given in the module notes: a collinear triplet
    # (routine in crystals) makes the cross product exactly zero. `- eps` puts y_k back at 0.
    y_k = torch.sqrt((cross**2).sum(dim=1) + (eps * length) ** 2) / length - eps

    zero = torch.zeros_like(length)
    planar = torch.stack(
        [
            torch.stack([length, zero], dim=-1),  # i, the receiving bond's far end
            torch.stack([zero, zero], dim=-1),  # j, the centre
            torch.stack([x_k, y_k], dim=-1),  # k
        ],
        dim=1,
    )
    planar = planar - planar.mean(dim=1, keepdim=True)
    return planar.to(config.float_th), atomic_numbers.to(config.float_th)


