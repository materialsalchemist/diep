"""Computing various graph based operations."""

from __future__ import annotations

import typing
import warnings

import dgl
import numpy as np
import torch

import diep

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def compute_pair_vector_and_distance(g: dgl.DGLGraph):
    """Calculate bond vectors and distances using dgl graphs.

    Args:
    g: DGL graph

    Returns:
    bond_vec (torch.tensor): bond distance between two atoms
    bond_dist (torch.tensor): vector from src node to dst node
    """
    dst_pos = g.ndata["pos"][g.edges()[1]] + g.edata["pbc_offshift"]
    src_pos = g.ndata["pos"][g.edges()[0]]
    bond_vec = dst_pos - src_pos
    bond_dist = torch.norm(bond_vec, dim=1)

    return bond_vec, bond_dist


def compute_theta_and_phi(edges: dgl.udf.EdgeBatch):
    """Calculate bond angle Theta and Phi using dgl graphs.

    Args:
    edges: DGL graph edges

    Returns:
    cos_theta: torch.Tensor
    phi: torch.Tensor
    triple_bond_lengths (torch.tensor):
    """
    angles = compute_theta(edges, cosine=True, directed=False)
    angles["phi"] = torch.zeros_like(angles["cos_theta"])
    return angles


def compute_theta(
    edges: dgl.udf.EdgeBatch, cosine: bool = False, directed: bool = True, eps=1e-7
) -> dict[str, torch.Tensor]:
    """User defined dgl function to calculate bond angles from edges in a graph.

    Args:
        edges: DGL graph edges
        cosine: Whether to return the cosine of the angle or the angle itself
        directed: Whether to the line graph was created with create directed line graph.
            In which case bonds (only those that are not self bonds) need to
            have their bond vectors flipped.
        eps: eps value used to clamp cosine values to avoid acos of values > 1.0

    Returns:
        dict[str, torch.Tensor]: Dictionary containing bond angles and distances
    """
    vec1 = edges.src["bond_vec"] * edges.src["src_bond_sign"] if directed else edges.src["bond_vec"]
    vec2 = edges.dst["bond_vec"]
    key = "cos_theta" if cosine else "theta"
    val = torch.sum(vec1 * vec2, dim=1) / (torch.norm(vec1, dim=1) * torch.norm(vec2, dim=1))
    val = val.clamp_(min=-1 + eps, max=1 - eps)  # stability for floating point numbers > 1.0
    if not cosine:
        val = torch.acos(val)
    return {key: val, "triple_bond_lengths": edges.dst["bond_dist"]}


def create_line_graph(
    g: dgl.DGLGraph,
    threebody_cutoff: float,
    directed: bool = False,
    error_handling: bool = False,
    numerical_noise: float = 1e-6,
) -> dgl.DGLGraph:
    """
    Calculate the three body indices from pair atom indices.

    Args:
        g: DGL graph
        threebody_cutoff (float): cutoff for three-body interactions
        directed (bool): Whether to create a directed line graph, or an M3gnet 3body line graph
            Default = False (M3Gnet)
        error_handling: whether to handle exception due to numerical error
            Default = False
        numerical_noise: a tiny noise added to lg construction to avoid numerical error
            Default = 1e-7

    Returns:
        l_g: DGL graph containing three body information from graph
    """

    def _build(cutoff: float) -> dgl.DGLGraph:
        # `graph_with_three_body` renumbers the surviving bonds 0..n_kept-1, so any line
        # graph built on top of it is in *pruned-bond* index space. Everything downstream
        # (three_cutoff, g.edges(), g.edata["bond_vec"], the three-body scatter width)
        # is in *parent-bond* index space, so the undirected line graph is translated
        # back before it leaves this function. See _remap_line_graph_to_bond_space.
        graph_with_three_body = prune_edges_by_features(
            g, feat_name="bond_dist", condition=lambda x: x > cutoff
        )
        if directed:
            # NOTE: the directed line graph is *not* remapped. It is unreachable from the
            # DIEP model (models/_diep.py always calls create_line_graph with the default
            # directed=False) and its consumer, compute_theta, only reads line-graph ndata,
            # which _create_directed_line_graph keeps self-consistent within pruned space.
            return _create_directed_line_graph(graph_with_three_body)
        return _remap_line_graph_to_bond_space(g, graph_with_three_body, _compute_3body(graph_with_three_body))

    if error_handling:
        try:
            return _build(threebody_cutoff)
        except Exception as e:
            # Print a warning if the first attempt fails
            warnings.warn(
                f"Initial line graph creation failed with error: {e}. "
                f"Adding numerical noise ({numerical_noise}) to threebody_cutoff and retrying.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _build(threebody_cutoff + numerical_noise)
    return _build(threebody_cutoff)


def _remap_line_graph_to_bond_space(
    graph: dgl.DGLGraph, pruned_graph: dgl.DGLGraph, line_graph: dgl.DGLGraph
) -> dgl.DGLGraph:
    """Translate an m3gnet-style line graph from pruned-bond into parent-bond index space.

    ``_compute_3body`` runs on the graph produced by ``prune_edges_by_features``, whose
    bonds are renumbered ``0 .. n_kept-1``. The resulting line graph's node ids are
    therefore *pruned-bond* ids. Every tensor those ids go on to index in the forward pass
    is built over the *parent* graph's bonds:

    * ``graph.edges()[1]`` and ``graph.edata["bond_vec"]`` (length ``num_bonds``),
    * ``polynomial_cutoff(g.edata["bond_dist"], threebody_cutoff)`` (length ``num_bonds``),
    * the three-body scatter, whose width is ``graph.num_edges()``.

    The two spaces coincide only when nothing is pruned (``threebody_cutoff == cutoff``).
    Under DIEP's shipped 5.0/4.0 configuration roughly half the bonds are pruned, so they
    diverge almost immediately. This function puts the line graph into parent-bond space so
    that the consuming layers -- whose arithmetic is correct as written -- see one index
    space throughout.

    Args:
        graph: the parent atom graph (all bonds).
        pruned_graph: the graph returned by ``prune_edges_by_features``; carries the
            ``edge_ids`` translation table from pruned-bond id to parent-bond id.
        line_graph: the line graph built over ``pruned_graph`` (pruned-bond index space).

    Returns:
        A line graph with exactly ``graph.num_edges()`` nodes, node ``i`` being parent bond
        ``i``, and ``n_triple_ij`` zero for every bond that participates in no triple.
    """
    num_bonds = graph.num_edges()

    # edge_ids[i] == parent-bond id of pruned bond i. Ascending by construction, since
    # prune_edges_by_features builds it with nonzero() on a boolean mask; the remap
    # therefore preserves the sorted-by-source-bond edge order that the three-body
    # scatter (get_segment_indices_from_n over n_triple_ij) depends on.
    edge_ids = pruned_graph.edata["edge_ids"].reshape(-1).long()

    lg_src, lg_dst = line_graph.edges()  # pruned-bond ids
    src = edge_ids[lg_src.long()].to(diep.int_th)  # parent-bond ids
    dst = edge_ids[lg_dst.long()].to(diep.int_th)  # parent-bond ids

    # Size explicitly from the parent bond count. Letting dgl infer the node count from
    # the largest id present is not robust: it would silently drop trailing bonds that
    # participate in no triple, and break the node-id offsets used by dgl.batch.
    remapped = dgl.graph((src, dst), num_nodes=num_bonds, device=graph.device)

    # One entry per parent bond, zero for bonds in no triple (including every bond
    # outside threebody_cutoff). Empty segments are why get_segment_indices_from_n had
    # to be made empty-segment correct.
    n_triple_ij = torch.zeros(num_bonds, dtype=diep.int_th, device=graph.device)
    n_nodes = line_graph.num_nodes()
    if n_nodes:
        n_triple_ij[edge_ids[:n_nodes]] = line_graph.ndata["n_triple_ij"].to(n_triple_ij.dtype)
    remapped.ndata["n_triple_ij"] = n_triple_ij

    # Line-graph node i is now parent bond i, so parent edge data transfers verbatim.
    for key in ("bond_dist", "bond_vec", "pbc_offset"):
        if key in graph.edata:
            remapped.ndata[key] = graph.edata[key]

    return remapped


def assert_lg_invariants(g: dgl.DGLGraph, lg: dgl.DGLGraph) -> None:
    """Assert that a three-body line graph is in parent-bond index space.

    The index-space defect this guards against produces no exception, no NaN and no
    anomalous loss curve -- training converges normally and test errors land in a healthy
    band -- so it has to be checked explicitly rather than waited for.

    Args:
        g: parent atom graph.
        lg: three-body line graph of ``g`` (undirected / m3gnet style).

    Raises:
        AssertionError: if any invariant is violated.
    """
    src = lg.edges()[0]
    n_triple = lg.ndata["n_triple_ij"]
    assert lg.num_nodes() == g.num_edges(), (
        f"line graph has {lg.num_nodes()} nodes but parent graph has {g.num_edges()} bonds; "
        "line graph node ids are not in parent-bond index space"
    )
    if src.numel():
        assert bool(torch.all(src[1:] >= src[:-1])), "line graph src not sorted ascending"
        assert int(src.max()) < g.num_edges(), "line graph src id out of range of parent bonds"
    expected = torch.bincount(src.long(), minlength=g.num_edges())
    assert torch.equal(expected.to(n_triple.dtype), n_triple), "n_triple_ij != bincount(line graph src)"
    assert int(n_triple.sum()) == int(src.numel()), "n_triple_ij.sum() != number of triples"


def ensure_line_graph_compatibility(
    graph: dgl.DGLGraph, line_graph: dgl.DGLGraph, threebody_cutoff: float, directed: bool = False, tol: float = 5e-6
) -> dgl.DGLGraph:
    """Ensure that line graph is compatible with graph.

    Sets edge data in line graph to be consistent with graph. The line graph is updated in place.

    Args:
        graph: atomistic graph
        line_graph: line graph of atomistic graph
        threebody_cutoff: cutoff for three-body interactions
        directed (bool): Whether to create a directed line graph, or an m3gnet 3body line graph (default: False, m3gnet)
        tol: numerical tolerance for cutoff
    """
    if directed:
        line_graph = _ensure_directed_line_graph_compatibility(graph, line_graph, threebody_cutoff, tol)
    else:
        line_graph = _ensure_3body_line_graph_compatibility(graph, line_graph, threebody_cutoff)

    return line_graph


def prune_edges_by_features(
    graph: dgl.DGLGraph,
    feat_name: str,
    condition: Callable[[torch.Tensor], torch.Tensor],
    keep_ndata: bool = False,
    keep_edata: bool = True,
    *args,
    **kwargs,
) -> dgl.DGLGraph:
    """Removes edges graph that do satisfy given condition based on a specified feature value.

    Returns a new graph with edges removed.

    Args:
        graph: DGL graph
        feat_name: edge field name
        condition: condition function. Must be a function where the first is the value
            of the edge field data and returns a Tensor of boolean values.
        keep_ndata: whether to keep node features
        keep_edata: whether to keep edge features
        *args: additional arguments to pass to condition function
        **kwargs: additional keyword arguments to pass to condition function

    Returns: dgl.Graph with removed edges.
    """
    if feat_name not in graph.edata:
        raise ValueError(f"Edge field {feat_name} not an edge feature in given graph.")

    valid_edges = torch.logical_not(condition(graph.edata[feat_name], *args, **kwargs))
    src, dst = graph.edges()
    src, dst = src[valid_edges], dst[valid_edges]
    e_ids = valid_edges.nonzero().squeeze()
    new_g = dgl.graph((src, dst), device=graph.device)
    new_g.edata["edge_ids"] = e_ids  # keep track of original edge ids

    if keep_ndata:
        for key, value in graph.ndata.items():
            new_g.ndata[key] = value
    if keep_edata:
        for key, value in graph.edata.items():
            new_g.edata[key] = value[valid_edges]

    return new_g


def _compute_3body(g: dgl.DGLGraph):
    """Calculate the three body indices from pair atom indices.

    Args:
        g: DGL graph

    Returns:
        l_g: DGL graph containing three body information from graph
        triple_bond_indices (np.ndarray): bond indices that form three-body
        n_triple_ij (np.ndarray): number of three-body angles for each bond
        n_triple_i (np.ndarray): number of three-body angles each atom
        n_triple_s (np.ndarray): number of three-body angles for each structure
    """
    n_atoms = g.num_nodes()
    first_col = g.edges()[0].cpu().numpy()

    # Count bonds per atom efficiently
    n_bond_per_atom = np.bincount(first_col, minlength=n_atoms)

    n_triple_i = n_bond_per_atom * (n_bond_per_atom - 1)
    n_triple = n_triple_i.sum()
    n_triple_ij = np.repeat(n_bond_per_atom - 1, n_bond_per_atom)

    triple_bond_indices = np.empty((n_triple, 2), dtype=diep.int_np)

    start = 0
    cs = 0
    for n in n_bond_per_atom:
        if n > 0:
            r = np.arange(n)
            x, y = np.meshgrid(r, r, indexing="xy")
            final = np.stack([y.ravel(), x.ravel()], axis=1)
            mask = final[:, 0] != final[:, 1]
            final = final[mask]
            triple_bond_indices[start : start + n * (n - 1)] = final + cs
            start += n * (n - 1)
            cs += n

    src_id = torch.tensor(triple_bond_indices[:, 0], dtype=diep.int_th)
    dst_id = torch.tensor(triple_bond_indices[:, 1], dtype=diep.int_th)
    l_g = dgl.graph((src_id, dst_id)).to(g.device)
    three_body_id = torch.cat(l_g.edges())
    n_triple_ij = torch.tensor(n_triple_ij, dtype=diep.int_th, device=g.device)  # type:ignore[assignment]

    max_three_body_id = three_body_id.max().item() + 1 if three_body_id.numel() > 0 else 0

    l_g.ndata["bond_dist"] = g.edata["bond_dist"][:max_three_body_id]  # type:ignore[misc]
    l_g.ndata["bond_vec"] = g.edata["bond_vec"][:max_three_body_id]  # type:ignore[misc]
    l_g.ndata["pbc_offset"] = g.edata["pbc_offset"][:max_three_body_id]  # type:ignore[misc]
    l_g.ndata["n_triple_ij"] = n_triple_ij[:max_three_body_id]  # type:ignore[misc]

    return l_g


def _create_directed_line_graph(
    graph: dgl.DGLGraph,
) -> dgl.DGLGraph:
    """Creates a line graph from a graph, considers periodic boundary conditions.

    Args:
        graph: DGL graph representing atom graph

    Returns:
        line_graph: DGL line graph of pruned graph to three body cutoff
    """
    with torch.no_grad():
        src_indices, dst_indices = graph.edges()
        images = graph.edata["pbc_offset"]
        all_indices = torch.arange(graph.number_of_nodes(), device=graph.device).unsqueeze(dim=0)
        num_bonds_per_atom = torch.count_nonzero(src_indices.unsqueeze(dim=1) == all_indices, dim=0)
        num_edges_per_bond = (num_bonds_per_atom - 1).repeat_interleave(num_bonds_per_atom)
        lg_src = torch.empty(num_edges_per_bond.sum(), dtype=diep.int_th, device=graph.device)  # type:ignore[call-overload]
        lg_dst = torch.empty(num_edges_per_bond.sum(), dtype=diep.int_th, device=graph.device)  # type:ignore[call-overload]

        incoming_edges = src_indices.unsqueeze(1) == dst_indices
        is_self_edge = src_indices == dst_indices
        not_self_edge = ~is_self_edge

        n = 0
        # create line graph edges for bonds that are self edges in atom graph
        if is_self_edge.any():
            edge_inds_s = is_self_edge.nonzero()
            lg_dst_s = edge_inds_s.repeat_interleave(num_edges_per_bond[is_self_edge] + 1)
            lg_src_s = incoming_edges[is_self_edge].nonzero()[:, 1].squeeze()
            lg_src_s = lg_src_s[lg_src_s != lg_dst_s]
            lg_dst_s = edge_inds_s.repeat_interleave(num_edges_per_bond[is_self_edge])
            n = len(lg_dst_s)
            lg_src[:n], lg_dst[:n] = lg_src_s, lg_dst_s

        # create line graph edges for bonds that are not self edges in atom graph
        shared_src = src_indices.unsqueeze(1) == src_indices
        back_tracking = (dst_indices.unsqueeze(1) == src_indices) & torch.all(-images.unsqueeze(1) == images, axis=2)  # type:ignore[call-overload]
        incoming = incoming_edges & (shared_src | ~back_tracking)

        edge_inds_ns = not_self_edge.nonzero().squeeze()
        lg_src_ns = incoming[not_self_edge].nonzero()[:, 1].squeeze()
        lg_dst_ns = edge_inds_ns.repeat_interleave(num_edges_per_bond[not_self_edge])
        lg_src[n:], lg_dst[n:] = lg_src_ns, lg_dst_ns
        lg = dgl.graph((lg_src, lg_dst))

        for key in graph.edata:
            lg.ndata[key] = graph.edata[key][: lg.number_of_nodes()]

        # we need to store the sign of bond vector when a bond is a src node in the line
        # graph in order to appropriately calculate angles when self edges are involved
        lg.ndata["src_bond_sign"] = torch.ones(
            (lg.number_of_nodes(), 1), dtype=lg.ndata["bond_vec"].dtype, device=lg.device
        )
        # if we flip self edges then we need to correct computed angles by pi - angle
        # lg.ndata["src_bond_sign"][edge_inds_s] = -lg.ndata["src_bond_sign"][edge_ind_s]
        # find the intersection for the rare cases where not all edges end up as nodes in the line graph
        all_ns, counts = torch.cat([torch.arange(lg.number_of_nodes(), device=graph.device), edge_inds_ns]).unique(
            return_counts=True
        )
        lg_inds_ns = all_ns[torch.where(counts > 1)]
        lg.ndata["src_bond_sign"][lg_inds_ns] = -lg.ndata["src_bond_sign"][lg_inds_ns]

    return lg


def _ensure_3body_line_graph_compatibility(graph: dgl.DGLGraph, line_graph: dgl.DGLGraph, threebody_cutoff: float):
    """Ensure that 3body line graph is compatible with a given graph.

    Sets node data in the line graph to be consistent with the graph's edge data. The line
    graph is updated in place.

    Line graphs produced by ``create_line_graph`` are in parent-bond index space: node ``i``
    of the line graph is bond ``i`` of ``graph``, and there are exactly ``graph.num_edges()``
    of them. The mapping is therefore the identity and the parent edge data transfers
    verbatim -- no ``threebody_cutoff`` comparison and no tolerance is involved. That also
    holds after ``dgl.batch``, because both graphs are batched in the same structure order.

    Args:
        graph: atomistic graph
        line_graph: line graph of atomistic graph
        threebody_cutoff: cutoff for three-body interactions. Unused; retained for API
            compatibility with _ensure_directed_line_graph_compatibility.
    """
    del threebody_cutoff  # unused: the pruned-space cutoff comparison is no longer needed

    if line_graph.num_nodes() != graph.num_edges():
        raise RuntimeError(
            "Line graph is not compatible with graph: expected one line-graph node per bond "
            f"({graph.num_edges()}), got {line_graph.num_nodes()}. Line graphs cached before "
            "the three-body index-space fix are in pruned-bond index space and cannot be "
            "reconciled; delete the cached line graphs and reprocess the dataset."
        )

    line_graph.ndata["bond_vec"] = graph.edata["bond_vec"]
    line_graph.ndata["bond_dist"] = graph.edata["bond_dist"]
    line_graph.ndata["pbc_offset"] = graph.edata["pbc_offset"]

    return line_graph


def _ensure_directed_line_graph_compatibility(
    graph: dgl.DGLGraph, line_graph: dgl.DGLGraph, threebody_cutoff: float, tol: float = 5e-6
) -> dgl.DGLGraph:
    """Ensure that line graph is compatible with graph.

    Sets edge data in line graph to be consistent with graph. The line graph is updated in place.

    Args:
        graph: atomistic graph
        line_graph: line graph of atomistic graph
        threebody_cutoff: cutoff for three-body interactions
        tol: numerical tolerance for cutoff
    """
    valid_edges = graph.edata["bond_dist"] <= threebody_cutoff

    # this means there probably is a bond that is just at the cutoff
    # this should only really occur when batching graphs
    if line_graph.number_of_nodes() > sum(valid_edges):
        valid_edges = graph.edata["bond_dist"] <= threebody_cutoff + tol

    # check again and raise if invalid
    if line_graph.number_of_nodes() > sum(valid_edges):
        raise RuntimeError("Line graph is not compatible with graph.")

    edge_ids = valid_edges.nonzero().squeeze()[: line_graph.number_of_nodes()]
    line_graph.ndata["edge_ids"] = edge_ids

    for key in graph.edata:
        line_graph.ndata[key] = graph.edata[key][edge_ids]

    src_indices, dst_indices = graph.edges()
    ns_edge_ids = (src_indices[edge_ids] != dst_indices[edge_ids]).nonzero().squeeze()
    line_graph.ndata["src_bond_sign"] = torch.ones(
        (line_graph.number_of_nodes(), 1), dtype=graph.edata["bond_vec"].dtype, device=line_graph.device
    )
    line_graph.ndata["src_bond_sign"][ns_edge_ids] = -line_graph.ndata["src_bond_sign"][ns_edge_ids]

    return line_graph
