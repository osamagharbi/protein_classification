"""DGL-native GVP encoder adapted from the local gvp-pytorch checkout.

This keeps the scalar/vector tuple formulation from the vendored upstream
implementation while avoiding the torch_geometric and torch_scatter runtime
dependencies that are not available in the graphmae environment.
"""

import functools

import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


SHREC25_SCALAR_INDICES = (0, 1)
SHREC25_NORMAL_INDICES = (2, 3, 4)
SHREC25_XYZ_INDICES = (5, 6, 7)


def tuple_sum(*args):
    return tuple(map(sum, zip(*args)))


def tuple_cat(*args, dim=-1):
    dim %= len(args[0][0].shape)
    s_args, v_args = list(zip(*args))
    return torch.cat(s_args, dim=dim), torch.cat(v_args, dim=dim)


def _norm_no_nan(x, axis=-1, keepdims=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), axis, keepdims), min=eps)
    return torch.sqrt(out) if sqrt else out


class GVP(nn.Module):
    def __init__(
        self,
        in_dims,
        out_dims,
        h_dim=None,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
    ):
        super().__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vector_gate = vector_gate

        if self.vi:
            self.h_dim = h_dim or max(self.vi, self.vo)
            self.wh = nn.Linear(self.vi, self.h_dim, bias=False)
            self.ws = nn.Linear(self.h_dim + self.si, self.so)
            if self.vo:
                self.wv = nn.Linear(self.h_dim, self.vo, bias=False)
                if self.vector_gate:
                    self.wsv = nn.Linear(self.so, self.vo)
        else:
            self.ws = nn.Linear(self.si, self.so)

        self.scalar_act, self.vector_act = activations
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if self.vi:
            s, v = x
            v = torch.transpose(v, -1, -2)
            vh = self.wh(v)
            vn = _norm_no_nan(vh, axis=-2)
            s = self.ws(torch.cat([s, vn], -1))
            if self.vo:
                v = self.wv(vh)
                v = torch.transpose(v, -1, -2)
                if self.vector_gate:
                    if self.vector_act:
                        gate = self.wsv(self.vector_act(s))
                    else:
                        gate = self.wsv(s)
                    v = v * torch.sigmoid(gate).unsqueeze(-1)
                elif self.vector_act:
                    v = v * self.vector_act(_norm_no_nan(v, axis=-1, keepdims=True))
        else:
            s = self.ws(x)
            if self.vo:
                v = torch.zeros(s.shape[0], self.vo, 3, device=self.dummy_param.device)

        if self.scalar_act:
            s = self.scalar_act(s)

        return (s, v) if self.vo else s


class _VDropout(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.drop_rate = drop_rate
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.bernoulli(
            (1 - self.drop_rate) * torch.ones(x.shape[:-1], device=self.dummy_param.device)
        ).unsqueeze(-1)
        return mask * x / (1 - self.drop_rate)


class Dropout(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()
        self.sdropout = nn.Dropout(drop_rate)
        self.vdropout = _VDropout(drop_rate)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            return self.sdropout(x)
        s, v = x
        return self.sdropout(s), self.vdropout(v)


class LayerNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.s, self.v = dims
        self.scalar_norm = nn.LayerNorm(self.s)

    def forward(self, x):
        if not self.v:
            return self.scalar_norm(x)
        s, v = x
        vn = _norm_no_nan(v, axis=-1, keepdims=True, sqrt=False)
        vn = torch.sqrt(torch.mean(vn, dim=-2, keepdim=True))
        return self.scalar_norm(s), v / vn.clamp(min=1e-8)


class DGLGVPConvLayer(nn.Module):
    def __init__(
        self,
        node_dims,
        edge_dims,
        n_message=3,
        n_feedforward=2,
        drop_rate=0.1,
        activations=(F.relu, torch.sigmoid),
        vector_gate=True,
    ):
        super().__init__()
        self.node_dims = node_dims
        self.edge_dims = edge_dims
        self.ns, self.nv = node_dims
        self.es, self.ev = edge_dims

        gvp_block = functools.partial(GVP, activations=activations, vector_gate=vector_gate)

        message_layers = []
        if n_message == 1:
            message_layers.append(
                gvp_block((2 * self.ns + self.es, 2 * self.nv + self.ev), node_dims, activations=(None, None))
            )
        else:
            message_layers.append(gvp_block((2 * self.ns + self.es, 2 * self.nv + self.ev), node_dims))
            for _ in range(n_message - 2):
                message_layers.append(gvp_block(node_dims, node_dims))
            message_layers.append(gvp_block(node_dims, node_dims, activations=(None, None)))
        self.message_func = nn.Sequential(*message_layers)

        ff_layers = []
        if n_feedforward == 1:
            ff_layers.append(gvp_block(node_dims, node_dims, activations=(None, None)))
        else:
            hid_dims = (4 * self.ns, 2 * self.nv)
            ff_layers.append(gvp_block(node_dims, hid_dims))
            for _ in range(n_feedforward - 2):
                ff_layers.append(gvp_block(hid_dims, hid_dims))
            ff_layers.append(gvp_block(hid_dims, node_dims, activations=(None, None)))
        self.ff_func = nn.Sequential(*ff_layers)

        self.norm = nn.ModuleList([LayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([Dropout(drop_rate) for _ in range(2)])

    def _edge_message(self, edges):
        src_s = edges.src["node_s"]
        dst_s = edges.dst["node_s"]
        src_v = edges.src["node_v"].view(-1, self.nv, 3)
        dst_v = edges.dst["node_v"].view(-1, self.nv, 3)
        edge_s = edges.data["edge_s"]
        edge_v = edges.data["edge_v"].view(-1, self.ev, 3)

        msg_s, msg_v = self.message_func(tuple_cat((src_s, src_v), (edge_s, edge_v), (dst_s, dst_v)))
        return {
            "msg_s": msg_s,
            "msg_v": msg_v.reshape(msg_v.shape[0], self.nv * 3),
        }

    def forward(self, g, x, edge_attr):
        with g.local_scope():
            g.ndata["node_s"] = x[0]
            g.ndata["node_v"] = x[1].reshape(x[1].shape[0], self.nv * 3)
            g.edata["edge_s"] = edge_attr[0]
            g.edata["edge_v"] = edge_attr[1].reshape(edge_attr[1].shape[0], self.ev * 3)

            g.apply_edges(self._edge_message)
            g.update_all(fn.copy_e("msg_s", "msg_s"), fn.mean("msg_s", "agg_s"))
            g.update_all(fn.copy_e("msg_v", "msg_v"), fn.mean("msg_v", "agg_v"))
            dh = (g.ndata["agg_s"], g.ndata["agg_v"].view(x[1].shape[0], self.nv, 3))

        x = self.norm[0](tuple_sum(x, self.dropout[0](dh)))
        dh = self.ff_func(x)
        x = self.norm[1](tuple_sum(x, self.dropout[1](dh)))
        return x


class DGLGVPEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden_dim,
        num_layers,
        dropout,
        feature_indices=None,
        num_hidden_vectors=8,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.feature_indices = tuple(feature_indices) if feature_indices is not None else None
        selected_indices = self.feature_indices if self.feature_indices is not None else tuple(range(in_dim))
        self.use_normal_vectors = set(SHREC25_NORMAL_INDICES).issubset(set(selected_indices))
        scalar_input_dim = self._count_scalar_dims(selected_indices)
        self.scalar_pad = scalar_input_dim == 0
        node_in_dims = (max(1, scalar_input_dim), 1 + int(self.use_normal_vectors))
        node_hidden_dims = (hidden_dim, num_hidden_vectors)
        edge_in_dims = (1, 1)
        edge_hidden_dims = (hidden_dim, 1)

        self.node_proj = nn.Sequential(
            LayerNorm(node_in_dims),
            GVP(node_in_dims, node_hidden_dims, activations=(None, None), vector_gate=True),
        )
        self.edge_proj = nn.Sequential(
            LayerNorm(edge_in_dims),
            GVP(edge_in_dims, edge_hidden_dims, activations=(None, None), vector_gate=True),
        )
        self.layers = nn.ModuleList(
            DGLGVPConvLayer(
                node_hidden_dims,
                edge_hidden_dims,
                drop_rate=dropout,
                activations=(F.relu, torch.sigmoid),
                vector_gate=True,
            )
            for _ in range(num_layers)
        )
        self.output_proj = nn.Sequential(
            LayerNorm(node_hidden_dims),
            GVP(node_hidden_dims, (hidden_dim, 0), activations=(None, None), vector_gate=False),
        )

    def _count_scalar_dims(self, selected_indices):
        excluded = set(SHREC25_XYZ_INDICES)
        if self.use_normal_vectors:
            excluded.update(SHREC25_NORMAL_INDICES)
        return sum(1 for idx in selected_indices if idx not in excluded)

    def _extract_positions(self, g, feat):
        if "pos" in g.ndata and torch.is_tensor(g.ndata["pos"]):
            return g.ndata["pos"].float()
        if feat.shape[1] >= SHREC25_XYZ_INDICES[-1] + 1:
            return feat[:, SHREC25_XYZ_INDICES].float()
        if feat.shape[1] == 3:
            return feat[:, :3].float()
        raise ValueError("GVP model requires 3D coordinates in g.ndata['pos'] or attr[:, 5:8]")

    def _split_node_inputs(self, feat, pos):
        selected_indices = self.feature_indices if self.feature_indices is not None else tuple(range(feat.shape[1]))
        scalar_indices = [idx for idx in selected_indices if idx not in set(SHREC25_XYZ_INDICES + SHREC25_NORMAL_INDICES)]
        if not self.use_normal_vectors:
            scalar_indices = [idx for idx in selected_indices if idx not in set(SHREC25_XYZ_INDICES)]

        if scalar_indices:
            scalar = feat[:, scalar_indices].float()
        else:
            scalar = pos.new_zeros(pos.shape[0], 1)

        vector_parts = [pos.unsqueeze(1)]
        if self.use_normal_vectors:
            if feat.shape[1] < SHREC25_NORMAL_INDICES[-1] + 1:
                raise ValueError("GVP normal-vector mode requires features 2:5 to be present")
            vector_parts.append(feat[:, SHREC25_NORMAL_INDICES].float().unsqueeze(1))

        vector = torch.cat(vector_parts, dim=1)
        return scalar, vector

    def _build_edge_inputs(self, g, pos):
        src, dst = g.edges()
        rel = pos[dst] - pos[src]
        dist = torch.norm(rel, dim=-1, keepdim=True).clamp(min=1e-8)
        direction = (rel / dist).unsqueeze(1)
        return dist, direction

    def forward(self, g, feat):
        pos = self._extract_positions(g, feat)
        node_scalar, node_vector = self._split_node_inputs(feat, pos)
        edge_scalar, edge_vector = self._build_edge_inputs(g, pos)

        node_state = self.node_proj((node_scalar, node_vector))
        edge_state = self.edge_proj((edge_scalar, edge_vector))
        for layer in self.layers:
            node_state = layer(g, node_state, edge_state)
        return self.output_proj(node_state)