"""PICE-GNN convolution layer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax
from torch_sparse import matmul


class PICEGNNConv(MessagePassing):
    """Alternating bus-branch convolution with PTDF-guided feature fusion."""

    def __init__(
        self,
        layer: int,
        in_channels,
        out_channels,
        ptdf: torch.Tensor,
        root_weight: bool = True,
        bias: bool = True,
        aggr: str = "add",
        attention_dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(aggr=aggr, **kwargs)
        if ptdf is None:
            raise ValueError("PICEGNNConv requires a PTDF matrix with shape [num_branches, num_buses].")
        if ptdf.dim() != 2:
            raise ValueError("ptdf must be a 2-D tensor with shape [num_branches, num_buses].")

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        if isinstance(out_channels, int):
            out_channels = (out_channels, out_channels)

        self.layer = layer
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.root_weight = root_weight
        self.attention_dropout = attention_dropout
        self.num_branches, self.num_buses = ptdf.shape
        self.register_buffer("ptdf", ptdf.float())

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.lin_bus_neigh = Linear(in_channels[0], out_channels[0], bias=bias)
        self.lin_bus_self = Linear(in_channels[0], out_channels[0], bias=bias) if root_weight else None
        self.lin_bus_for_branch = Linear(in_channels[0], out_channels[0], bias=bias)

        self.lin_branch_neigh = Linear(in_channels[1], out_channels[2], bias=bias)
        self.lin_branch_self = Linear(in_channels[1], out_channels[2], bias=bias) if root_weight else None
        self.lin_branch_for_bus = Linear(in_channels[1], out_channels[2], bias=bias)

        self.lin_attention = Linear(in_channels[layer % 2], in_channels[layer % 2], bias=True)
        self.attention_vector = nn.Parameter(torch.empty(2 * in_channels[layer % 2], 1))

        self.ptdf_column_weight = nn.Parameter(torch.empty(1, self.num_buses))
        self.ptdf_matrix_weight = nn.Parameter(torch.empty(self.num_branches, self.num_buses))
        if self.layer % 2 == 0:
            self.lin_cross = Linear(out_channels[2], out_channels[1], bias=bias)
            self.lin_ptdf_score = Linear(self.num_buses, 1, bias=True)
        else:
            self.lin_cross = Linear(out_channels[0], out_channels[3], bias=bias)
            self.lin_ptdf_score = Linear(self.num_branches, 1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.lin_bus_neigh.reset_parameters()
        self.lin_bus_for_branch.reset_parameters()
        self.lin_branch_neigh.reset_parameters()
        self.lin_branch_for_bus.reset_parameters()
        self.lin_attention.reset_parameters()
        self.lin_cross.reset_parameters()
        self.lin_ptdf_score.reset_parameters()
        if self.lin_bus_self is not None:
            self.lin_bus_self.reset_parameters()
        if self.lin_branch_self is not None:
            self.lin_branch_self.reset_parameters()
        nn.init.xavier_uniform_(self.attention_vector)
        nn.init.xavier_uniform_(self.ptdf_column_weight)
        nn.init.xavier_uniform_(self.ptdf_matrix_weight)

    def forward(self, x_bus, x_branch, edge_index, node_edge_index, line_graph_edge_index):
        if isinstance(x_bus, Tensor):
            x_bus = (x_bus, x_bus)
        if isinstance(x_branch, Tensor):
            x_branch = (x_branch, x_branch)

        if self.layer % 2 == 0:
            bus_msg = self.propagate(edge_index, x=x_bus[0], use_ptdf=False)
            out_bus = self.lin_bus_neigh(bus_msg)
            if self.lin_bus_self is not None:
                out_bus = out_bus + self.lin_bus_self(x_bus[1])

            branch_msg = self.lin_branch_for_bus(x_branch[0])
            branch_to_bus = self.propagate(
                node_edge_index.flip([0]),
                x=branch_msg,
                size=(x_branch[0].shape[0], x_bus[0].shape[0]),
                use_ptdf=True,
            )
            out_bus = torch.cat([out_bus, self.lin_cross(branch_to_bus)], dim=1)
            return out_bus, x_branch[0]

        branch_msg = self.propagate(line_graph_edge_index, x=x_branch[0], use_ptdf=False)
        out_branch = self.lin_branch_neigh(branch_msg)
        if self.lin_branch_self is not None:
            out_branch = out_branch + self.lin_branch_self(x_branch[1])

        bus_msg = self.lin_bus_for_branch(x_bus[0])
        bus_to_branch = self.propagate(
            node_edge_index,
            x=bus_msg,
            size=(x_bus[0].shape[0], x_branch[0].shape[0]),
            use_ptdf=True,
        )
        out_branch = torch.cat([out_branch, self.lin_cross(bus_to_branch)], dim=1)
        return x_bus[0], out_branch

    def message(self, x_j: Tensor, x_i: Tensor, edge_index: Tensor, use_ptdf: bool = False) -> Tensor:
        if not use_ptdf:
            x_i_att = self.lin_attention(x_i)
            x_j_att = self.lin_attention(x_j)
            score = torch.matmul(torch.cat([x_i_att, x_j_att], dim=-1), self.attention_vector)
            score = self.leaky_relu(score)
            alpha = softmax(score, edge_index[1])
            alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)
            return x_j_att * alpha

        ptdf = self.ptdf.to(x_j.device)
        ptdf = ptdf * self.ptdf_column_weight.to(x_j.device)
        ptdf = self.leaky_relu(ptdf)
        ptdf = ptdf * self.ptdf_matrix_weight.to(x_j.device)

        if self.layer % 2 == 0:
            scores = self.lin_ptdf_score(ptdf)
            row_indices = edge_index[0] % self.num_branches
        else:
            scores = self.lin_ptdf_score(ptdf.t())
            row_indices = edge_index[0] % self.num_buses

        alpha = softmax(scores[row_indices], edge_index[1])
        alpha = F.dropout(alpha, p=self.attention_dropout, training=self.training)
        return x_j * alpha

    def message_and_aggregate(self, adj_t: Tensor, x):  # noqa: D102
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels}, aggr={self.aggr})"


# Backward-compatible alias for older scripts/checkpoints.
NEA_GNNConv = PICEGNNConv
