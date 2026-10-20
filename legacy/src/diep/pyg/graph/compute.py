"""Graph operations for the PyG backend: bond geometry and the three-body line graph.

Index spaces, stated up front, because confusing them is exactly the defect this backend
was written not to reproduce:

* **parent-bond space** -- ids into ``data.edge_index``, of which there are
  ``data.num_edges``. Everything the model indexes lives here: ``edge_index`` itself,
  ``bond_vec``, ``bond_dist``, the polynomial three-body cutoff, and the width of the
  three-body scatter.
* **pruned-bond space** -- ids into the subset of bonds shorter than ``threebody_cutoff``,
  renumbered ``0 .. n_kept-1``.

``create_line_graph`` builds ``triple_index`` directly in parent-bond space. There is no
intermediate pruned graph to leak ids from, and :func:`assert_lg_invariants` pins the
result down. The DGL side reaches the same place by remapping through the ``edge_ids``
table that ``prune_edges_by_features`` records.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch_geometric.data import Data

import diep

if TYPE_CHECKING:
    from collections.abc import Callable


class DIEPData(Data):
    """A structure's atom graph and its three-body line graph in a single PyG ``Data``.

    Node-level: ``node_type``, ``frac_coords``, ``pos``.
    Edge-level (parent-bond space): ``edge_index`` (2, E), ``pbc_offset``, ``pbc_offshift``,
    ``bond_vec``, ``bond_dist``, ``n_triple_ij`` (E,).
    Triple-level: ``triple_index`` (2, T), holding *bond* ids, not atom ids.
    Graph-level: ``lattice`` (1, 3, 3), ``state_attr``.

    ``__inc__`` is overridden so that batching offsets ``triple_index`` by the running bond
    count rather than the node count -- the line graph's nodes are bonds. That is the PyG
    analogue of the node-id offsets ``dgl.batch`` applies to batched line graphs.
    """

    def __inc__(self, key: str, value, *args, **kwargs):  # noqa: D105
        if key == "triple_index":
            return self.num_edges
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key: str, value, *args, **kwargs):  # noqa: D105
        if key == "lattice":
            return 0  # (1, 3, 3) per structure -> (B, 3, 3) per batch
        if key == "n_triple_ij":
            return 0  # one entry per bond, concatenated like any other edge attribute
        return super().__cat_dim__(key, value, *args, **kwargs)

    @property
    def num_bonds(self) -> int:
        """Number of bonds, i.e. the size of the parent-bond index space."""
        return int(self.edge_index.size(1))

    @property
    def num_triples(self) -> int:
        """Number of three-body triples."""
        return 0 if self.triple_index is None else int(self.triple_index.size(1))


def compute_pair_vector_and_distance(data: Data) -> tuple[torch.Tensor, torch.Tensor]:
    """Calculate bond vectors and distances.

    Args:
        data: graph carrying ``pos``, ``edge_index`` and ``pbc_offshift``.

    Returns:
        bond_vec: vector from src atom to dst atom, including the periodic image shift.
        bond_dist: length of each bond vector.
    """
    src, dst = data.edge_index
    dst_pos = data.pos[dst] + data.pbc_offshift
    src_pos = data.pos[src]
    bond_vec = dst_pos - src_pos
    return bond_vec, torch.norm(bond_vec, dim=1)


def prune_edges_by_features(
    data: Data,
    feat_name: str,
    condition: Callable[[torch.Tensor], torch.Tensor],
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the bonds that do *not* satisfy ``condition``, and their parent-bond ids.

    Unlike the DGL version this does not build a renumbered graph -- returning the surviving
    parent-bond ids directly removes the opportunity to confuse the two index spaces.

    Args:
        data: graph to prune.
        feat_name: name of the edge attribute the condition is applied to.
        condition: function of the edge attribute returning a boolean mask of edges to drop.
        *args: extra positional arguments forwarded to ``condition``.
        **kwargs: extra keyword arguments forwarded to ``condition``.

    Returns:
        edge_index: the surviving bonds' ``(2, n_kept)`` index tensor.
        edge_ids: their parent-bond ids, ascending.
    """
    value = getattr(data, feat_name, None)
    if value is None:
        raise ValueError(f"Edge field {feat_name} not an edge feature in given graph.")
    keep = torch.logical_not(condition(value, *args, **kwargs))
    edge_ids = keep.nonzero(as_tuple=False).reshape(-1)
    return data.edge_index[:, edge_ids], edge_ids


def _triples_from_bonds(
    src: torch.Tensor, edge_ids: torch.Tensor, num_nodes: int, num_bonds: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate every ordered pair of distinct bonds that share a source atom.

    Args:
        src: source atom of each *kept* bond, in kept order.
        edge_ids: parent-bond id of each kept bond, ascending.
        num_nodes: number of atoms.
        num_bonds: number of bonds in the parent graph.

    Returns:
        triple_index: ``(2, T)`` pairs of *parent-bond* ids sharing a centre atom, with the
            first row ascending.
        n_triple_ij: ``(num_bonds,)`` count of triples per parent bond, zero for bonds that
            are pruned or whose centre atom has only one bond.

    The pair ordering matches the DGL implementation: for a centre atom whose bonds are
    ``b_0 < b_1 < ... < b_{n-1}``, the pairs are emitted as
    ``(b_0, b_1), (b_0, b_2), ..., (b_1, b_0), (b_1, b_2), ...`` -- grouped by first bond,
    partners ascending. Unlike the DGL version this does not assume the bonds are already
    grouped by source atom in the edge list.
    """
    device = src.device
    n_kept = int(src.numel())
    zeros_e = torch.zeros(num_bonds, dtype=diep.int_th, device=device)
    if n_kept == 0:
        return torch.zeros((2, 0), dtype=torch.long, device=device), zeros_e

    bonds_per_atom = torch.bincount(src, minlength=num_nodes)
    # `perm` lists kept-bond positions grouped by centre atom, ascending within each group.
    perm = torch.argsort(src, stable=True)
    group_start = torch.cumsum(bonds_per_atom, 0) - bonds_per_atom  # exclusive prefix sum
    rank = torch.empty(n_kept, dtype=torch.long, device=device)
    rank[perm] = torch.arange(n_kept, dtype=torch.long, device=device)
    rank_in_group = rank - group_start[src]  # position of each bond among its atom's bonds

    n_partners = (bonds_per_atom - 1).clamp_min(0)[src]  # triples per kept bond
    first_kept = torch.repeat_interleave(torch.arange(n_kept, dtype=torch.long, device=device), n_partners)
    # offset 0..n_partners-1 within each first-bond group
    offset = torch.arange(int(n_partners.sum()), dtype=torch.long, device=device) - torch.repeat_interleave(
        torch.cumsum(n_partners, 0) - n_partners, n_partners
    )
    # skip the bond itself: partners are the group's bonds with self removed
    own_rank = rank_in_group[first_kept]
    partner_rank = offset + (offset >= own_rank).long()
    second_kept = perm[group_start[src[first_kept]] + partner_rank]

    # translate kept-bond ids into parent-bond ids; edge_ids is ascending, so first stays sorted
    triple_index = torch.stack([edge_ids[first_kept], edge_ids[second_kept]], dim=0)

    n_triple_ij = zeros_e.clone()
    n_triple_ij[edge_ids] = n_partners.to(n_triple_ij.dtype)
    return triple_index, n_triple_ij


def create_line_graph(data: Data, threebody_cutoff: float, in_place: bool = True) -> Data:
    """Build the three-body line graph, in parent-bond index space.

    Args:
        data: graph carrying ``edge_index`` and ``bond_dist``.
        threebody_cutoff: bonds longer than this take part in no triple.
        in_place: write ``triple_index`` and ``n_triple_ij`` onto ``data`` and return it.
            When False, returns a shallow copy instead.

    Returns:
        The graph, with ``triple_index`` (2, T) holding parent-bond ids and ``n_triple_ij``
        (num_bonds,) holding the triple count of each bond -- zero for bonds in no triple.
    """
    src, _ = data.edge_index
    num_bonds = int(data.edge_index.size(1))
    num_nodes = int(data.num_nodes)

    _, edge_ids = prune_edges_by_features(
        data, feat_name="bond_dist", condition=lambda x: x > threebody_cutoff
    )
    triple_index, n_triple_ij = _triples_from_bonds(src[edge_ids], edge_ids, num_nodes, num_bonds)

    out = data if in_place else data.clone()
    out.triple_index = triple_index
    out.n_triple_ij = n_triple_ij
    return out


def ensure_line_graph_compatibility(data: Data, threebody_cutoff: float) -> Data:
    """Check a cached line graph against a freshly built graph, rebuilding if absent.

    The PyG line graph is two tensors stored on the graph itself, so there is no separate
    object to reconcile: node data cannot go stale the way it can on a DGL line graph, whose
    ``bond_vec`` / ``bond_dist`` / ``pbc_offset`` node features are copies of the parent's
    edge data. All that is left to check is that the cached ``triple_index`` is sized against
    this graph's bond count.

    Args:
        data: graph, possibly already carrying ``triple_index`` and ``n_triple_ij``.
        threebody_cutoff: cutoff for three-body interactions.

    Returns:
        The graph with a valid line graph attached.
    """
    if getattr(data, "triple_index", None) is None or getattr(data, "n_triple_ij", None) is None:
        return create_line_graph(data, threebody_cutoff)
    num_bonds = int(data.edge_index.size(1))
    if int(data.n_triple_ij.numel()) != num_bonds:
        raise RuntimeError(
            f"Cached line graph is not compatible with graph: n_triple_ij has "
            f"{int(data.n_triple_ij.numel())} entries but the graph has {num_bonds} bonds. "
            "Cached line graphs must be in parent-bond index space; regenerate the dataset."
        )
    if int(data.triple_index.numel()) and int(data.triple_index.max()) >= num_bonds:
        raise RuntimeError(
            "Cached line graph references bond ids outside this graph; regenerate the dataset."
        )
    return data


def assert_lg_invariants(data: Data) -> None:
    """Assert that the three-body line graph is in parent-bond index space.

    The index-space defect this guards against produces no exception, no NaN and no
    anomalous loss curve -- training converges normally and test errors land in a healthy
    band -- so it has to be checked explicitly rather than waited for.

    Args:
        data: graph carrying ``edge_index``, ``triple_index`` and ``n_triple_ij``.

    Raises:
        AssertionError: if any invariant is violated.
    """
    num_bonds = int(data.edge_index.size(1))
    src = data.triple_index[0]
    n_triple = data.n_triple_ij
    assert int(n_triple.numel()) == num_bonds, (
        f"n_triple_ij has {int(n_triple.numel())} entries but the graph has {num_bonds} bonds; "
        "the line graph is not in parent-bond index space"
    )
    if src.numel():
        assert bool(torch.all(src[1:] >= src[:-1])), "line graph src not sorted ascending"
        assert int(src.max()) < num_bonds, "line graph src id out of range of parent bonds"
        assert int(data.triple_index[1].max()) < num_bonds, "line graph dst id out of range of parent bonds"
    expected = torch.bincount(src.long(), minlength=num_bonds)
    assert torch.equal(expected.to(n_triple.dtype), n_triple), "n_triple_ij != bincount(line graph src)"
    assert int(n_triple.sum()) == int(src.numel()), "n_triple_ij.sum() != number of triples"


def compute_theta(
    data: Data, cosine: bool = False, eps: float = 1e-7
) -> dict[str, torch.Tensor]:
    """Bond angles for every triple.

    Args:
        data: graph carrying ``triple_index``, ``bond_vec`` and ``bond_dist``.
        cosine: return the cosine of the angle rather than the angle itself.
        eps: clamp applied to the cosine before ``acos`` for floating-point stability.

    Returns:
        A dict with the angle (key ``cos_theta`` or ``theta``) and ``triple_bond_lengths``.
    """
    first, second = data.triple_index
    vec1 = data.bond_vec[first]
    vec2 = data.bond_vec[second]
    val = torch.sum(vec1 * vec2, dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1))
    val = val.clamp_(min=-1 + eps, max=1 - eps)
    key = "cos_theta" if cosine else "theta"
    if not cosine:
        val = torch.acos(val)
    return {key: val, "triple_bond_lengths": data.bond_dist[second]}
