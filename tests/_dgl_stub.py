"""Minimal stand-in for the slice of the DGL API that DIEP's three-body path uses.

DGL publishes no aarch64 / torch-2.11 wheel, so the real library cannot be installed on
this machine. This shim implements exactly the operations the line-graph construction,
ThreeBodyInteractions and DIEPIntegrator touch, with DGL's documented semantics:

  * dgl.graph((u, v), num_nodes=None, device=None) keeps edges in insertion order and
    infers num_nodes from max(id)+1 when not given.
  * ndata / edata are per-node / per-edge tensor dicts with first-dim validation.
  * dgl.batch concatenates graphs, offsetting node ids by the running node count, and
    concatenates ndata / edata in the same order.

It is NOT a general DGL replacement. Message passing, readouts and dataloading raise.
"""
from __future__ import annotations

import contextlib
import sys
import types

import torch


class _DataDict(dict):
    def __init__(self, graph, kind):
        super().__init__()
        self._graph = graph
        self._kind = kind

    def _expected(self):
        return self._graph.num_nodes() if self._kind == "node" else self._graph.num_edges()

    def __setitem__(self, key, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        exp = self._expected()
        if value.shape[0] != exp:
            raise RuntimeError(
                f"Expect number of {self._kind}s to be {exp}, but got {value.shape[0]} for '{key}'"
            )
        super().__setitem__(key, value.to(self._graph.device))


class DGLGraph:
    def __init__(self, src, dst, num_nodes=None, device=None):
        src = torch.as_tensor(src)
        dst = torch.as_tensor(dst)
        if src.numel() == 0:
            src = src.reshape(0).to(torch.int64 if src.dtype not in (torch.int32, torch.int64) else src.dtype)
            dst = dst.reshape(0).to(src.dtype)
        if src.dtype not in (torch.int32, torch.int64):
            src = src.long()
        if dst.dtype != src.dtype:
            dst = dst.to(src.dtype)
        if device is None:
            device = src.device if src.numel() else torch.device("cpu")
        device = torch.device(device)
        self._device = device
        self._src = src.to(device)
        self._dst = dst.to(device)
        if num_nodes is None:
            if self._src.numel() == 0:
                num_nodes = 0
            else:
                num_nodes = int(max(int(self._src.max()), int(self._dst.max()))) + 1
        self._num_nodes = int(num_nodes)
        self.ndata = _DataDict(self, "node")
        self.edata = _DataDict(self, "edge")
        self._batch_num_nodes = None
        self._batch_num_edges = None

    # --- topology -------------------------------------------------------
    def edges(self, form="uv", order="eid"):
        return self._src, self._dst

    def num_nodes(self):
        return self._num_nodes

    def num_edges(self):
        return int(self._src.numel())

    number_of_nodes = num_nodes
    number_of_edges = num_edges

    @property
    def idtype(self):
        return self._src.dtype

    @property
    def device(self):
        return self._device

    def batch_num_nodes(self):
        if self._batch_num_nodes is None:
            return torch.tensor([self._num_nodes])
        return self._batch_num_nodes

    def batch_num_edges(self):
        if self._batch_num_edges is None:
            return torch.tensor([self.num_edges()])
        return self._batch_num_edges

    def to(self, device):
        device = torch.device(device)
        new = DGLGraph(self._src, self._dst, num_nodes=self._num_nodes, device=device)
        for k, v in self.ndata.items():
            new.ndata[k] = v.to(device)
        for k, v in self.edata.items():
            new.edata[k] = v.to(device)
        new._batch_num_nodes = self._batch_num_nodes
        new._batch_num_edges = self._batch_num_edges
        return new

    def __repr__(self):
        return f"DGLGraph(num_nodes={self._num_nodes}, num_edges={self.num_edges()})"

    def apply_edges(self, func):
        """Run a DGL edge UDF and store its output in edata."""
        out = func(_EdgeBatch(self))
        for key, value in out.items():
            self.edata[key] = value

    def update_all(self, message_func, reduce_func):
        """Message passing: build a per-edge message, aggregate it at the DESTINATION node."""
        if message_func.kind == "copy_e":
            msg = self.edata[message_func.src_field]
        elif message_func.kind == "copy_u":
            msg = self.ndata[message_func.src_field][self._src.long()]
        else:
            raise NotImplementedError(f"dgl shim: message func {message_func.kind}")
        out = torch.zeros(
            (self._num_nodes, *msg.shape[1:]), dtype=msg.dtype, device=msg.device
        )
        index = self._dst.long()
        if reduce_func.op == "sum":
            out.index_add_(0, index, msg)
        elif reduce_func.op == "mean":
            out.index_add_(0, index, msg)
            counts = torch.bincount(index, minlength=self._num_nodes).clamp_min(1)
            out = out / counts.view(-1, *([1] * (msg.dim() - 1))).to(out.dtype)
        else:
            raise NotImplementedError(f"dgl shim: reduce op {reduce_func.op}")
        self.ndata[reduce_func.out_field] = out

    @contextlib.contextmanager
    def local_scope(self):
        """Snapshot ndata/edata, restore on exit (dgl.DGLGraph.local_scope)."""
        saved_n, saved_e = dict(self.ndata), dict(self.edata)
        try:
            yield self
        finally:
            self.ndata.clear()
            self.ndata.update(saved_n)
            self.edata.clear()
            self.edata.update(saved_e)

    @property
    def batch_size(self):
        return int(self.batch_num_nodes().numel())


class _EdgeBatch:
    """Stand-in for dgl.udf.EdgeBatch: per-edge views of src/dst node data and edge data."""

    def __init__(self, graph):
        src, dst = graph.edges()
        self.src = {k: v[src.long()] for k, v in graph.ndata.items()}
        self.dst = {k: v[dst.long()] for k, v in graph.ndata.items()}
        self.data = dict(graph.edata)


Graph = DGLGraph


def graph(data, num_nodes=None, idtype=None, device=None):
    src, dst = data
    return DGLGraph(src, dst, num_nodes=num_nodes, device=device)


def batch(graphs):
    graphs = list(graphs)
    if not graphs:
        raise ValueError("empty batch")
    device = graphs[0].device
    srcs, dsts, offset = [], [], 0
    for g in graphs:
        s, d = g.edges()
        srcs.append(s.long() + offset)
        dsts.append(d.long() + offset)
        offset += g.num_nodes()
    idtype = graphs[0].idtype
    out = DGLGraph(
        torch.cat(srcs).to(idtype) if srcs else torch.zeros(0, dtype=idtype),
        torch.cat(dsts).to(idtype) if dsts else torch.zeros(0, dtype=idtype),
        num_nodes=offset,
        device=device,
    )
    for key in graphs[0].ndata:
        out.ndata[key] = torch.cat([g.ndata[key] for g in graphs], dim=0)
    for key in graphs[0].edata:
        out.edata[key] = torch.cat([g.edata[key] for g in graphs], dim=0)
    out._batch_num_nodes = torch.tensor([g.num_nodes() for g in graphs])
    out._batch_num_edges = torch.tensor([g.num_edges() for g in graphs])
    return out


class _BuiltinFunc:
    """Stand-in for dgl.function message/reduce builtins."""

    def __init__(self, kind, src_field=None, out_field=None, op=None):
        self.kind = kind
        self.src_field = src_field
        self.out_field = out_field
        self.op = op


def _node_batch_index(graph):
    """Structure index of each node, from the stored batch_num_nodes."""
    counts = graph.batch_num_nodes().to(graph.device)
    return torch.repeat_interleave(torch.arange(counts.numel(), device=graph.device), counts)


def _edge_batch_index(graph):
    counts = graph.batch_num_edges().to(graph.device)
    return torch.repeat_interleave(torch.arange(counts.numel(), device=graph.device), counts)


def _readout(graph, feat, op, per_node=True, weight=None):
    index = _node_batch_index(graph) if per_node else _edge_batch_index(graph)
    values = graph.ndata[feat] if per_node else graph.edata[feat]
    if weight is not None:
        w = graph.ndata[weight] if per_node else graph.edata[weight]
        values = values * w
    size = graph.batch_size
    out = torch.zeros((size, *values.shape[1:]), dtype=values.dtype, device=values.device)
    out.index_add_(0, index, values)
    if op == "mean":
        counts = torch.bincount(index, minlength=size).clamp_min(1)
        out = out / counts.view(-1, *([1] * (values.dim() - 1))).to(out.dtype)
    elif op != "sum":
        raise NotImplementedError(f"dgl shim: readout op {op}")
    return out


def _unimplemented(name):
    def _f(*a, **k):
        raise NotImplementedError(f"dgl shim: {name} not implemented")
    return _f


def install():
    """Register the shim as `dgl` (and the submodules DIEP imports) in sys.modules."""
    if "dgl" in sys.modules and getattr(sys.modules["dgl"], "_is_shim", False):
        return sys.modules["dgl"]

    dgl = types.ModuleType("dgl")
    dgl._is_shim = True
    dgl.DGLGraph = DGLGraph
    dgl.Graph = Graph
    dgl.graph = graph
    dgl.batch = batch
    dgl.NID = "_ID"
    dgl.EID = "_ID"
    dgl.readout_nodes = lambda g, feat, op="sum", **k: _readout(g, feat, op, per_node=True)
    dgl.readout_edges = lambda g, feat, op="sum", **k: _readout(g, feat, op, per_node=False)
    dgl.sum_nodes = lambda g, feat, weight=None, **k: _readout(g, feat, "sum", True, weight)
    dgl.sum_edges = lambda g, feat, weight=None, **k: _readout(g, feat, "sum", False, weight)
    dgl.mean_nodes = lambda g, feat, weight=None, **k: _readout(g, feat, "mean", True, weight)
    dgl.broadcast_nodes = lambda g, x, **k: x[_node_batch_index(g)]
    dgl.broadcast_edges = lambda g, x, **k: x[_edge_batch_index(g)]
    for n in ("softmax_nodes", "softmax_edges", "node_subgraph", "unbatch"):
        setattr(dgl, n, _unimplemented(n))

    udf = types.ModuleType("dgl.udf")
    udf.EdgeBatch = object
    udf.NodeBatch = object
    dgl.udf = udf

    function = types.ModuleType("dgl.function")
    function.copy_e = lambda src, out: _BuiltinFunc("copy_e", src_field=src, out_field=out)
    function.copy_u = lambda src, out: _BuiltinFunc("copy_u", src_field=src, out_field=out)
    function.sum = lambda msg, out: _BuiltinFunc("reduce", src_field=msg, out_field=out, op="sum")
    function.mean = lambda msg, out: _BuiltinFunc("reduce", src_field=msg, out_field=out, op="mean")
    dgl.function = function

    data = types.ModuleType("dgl.data")
    data_utils = types.ModuleType("dgl.data.utils")

    class Subset:
        def __init__(self, dataset, indices):
            self.dataset, self.indices = dataset, indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, i):
            return self.dataset[self.indices[i]]

    data_utils.Subset = Subset
    data_utils.save_graphs = _unimplemented("save_graphs")
    data_utils.load_graphs = _unimplemented("load_graphs")
    data_utils.split_dataset = _unimplemented("split_dataset")
    data.utils = data_utils
    data.Subset = Subset
    data.DGLDataset = object
    dgl.data = data

    dataloading = types.ModuleType("dgl.dataloading")

    class GraphDataLoader:
        def __init__(self, *a, **k):
            raise NotImplementedError("dgl shim: GraphDataLoader not implemented")

    dataloading.GraphDataLoader = GraphDataLoader
    dgl.dataloading = dataloading

    nn = types.ModuleType("dgl.nn")

    class _StubModule:
        def __init__(self, *a, **k):
            raise NotImplementedError("dgl shim: dgl.nn module not implemented")

    def _nn_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (_StubModule,), {})

    nn.__getattr__ = _nn_getattr
    dgl.nn = nn

    def _dgl_getattr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _unimplemented(name)

    dgl.__getattr__ = _dgl_getattr

    for name, mod in [
        ("dgl", dgl), ("dgl.udf", udf), ("dgl.function", function),
        ("dgl.data", data), ("dgl.data.utils", data_utils),
        ("dgl.dataloading", dataloading), ("dgl.nn", nn),
    ]:
        sys.modules[name] = mod
    return dgl
