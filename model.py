"""Model definitions for PICE-GNN and baseline GNNs."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.nn import Dropout, ReLU
from torch_geometric.nn import GATConv, GCNConv, GINConv, MLP, SAGEConv
from torch_geometric.utils import to_undirected

from layer import PICEGNNConv


class Classification(torch.nn.Module):
    """Two-class prediction head."""

    def __init__(self, num_layers: int, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = ReLU(inplace=True)

        for layer_id in range(num_layers):
            layer_in = in_channels if layer_id == 0 else hidden_channels
            layer_out = out_channels if layer_id == num_layers - 1 else hidden_channels
            self.layers.append(torch.nn.Linear(layer_in, layer_out))

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() == 2:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x):
        for layer_id, layer in enumerate(self.layers):
            x = layer(x)
            if layer_id != len(self.layers) - 1:
                x = self.activation(x)
        return F.softmax(x, dim=1)


class PICE_GNN(torch.nn.Module):
    """Physics-Informed Co-Embedding Graph Neural Network."""

    def __init__(self, num_layers, dropout, in_channels, hidden_channels, out_channels, ptdf):
        super().__init__()
        self.name = "PICE_GNN"
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = Dropout(p=dropout)
        self.activation = ReLU(inplace=True)

        current_channels = in_channels
        for layer_id in range(num_layers):
            if layer_id > 0:
                if layer_id % 2 == 0:
                    current_channels = [hidden_channels[0], hidden_channels[2] + hidden_channels[3]]
                else:
                    current_channels = [hidden_channels[0] + hidden_channels[1], hidden_channels[2]]

            layer_out_channels = out_channels if layer_id == num_layers - 1 else hidden_channels
            self.convs.append(
                PICEGNNConv(
                    layer=layer_id,
                    in_channels=current_channels,
                    out_channels=layer_out_channels,
                    ptdf=ptdf,
                    normalize=False,
                    root_weight=True,
                )
            )

        self.reset_parameters()

    def reset_parameters(self):
        for param in self.parameters():
            if param.dim() == 2:
                torch.nn.init.xavier_uniform_(param)

    def forward(self, x_bus, x_branch, edge_index, node_edge_index, line_graph_edge_index):
        for conv in self.convs:
            x_bus, x_branch = conv(x_bus, x_branch, edge_index, node_edge_index, line_graph_edge_index)
            x_bus = self.dropout(self.activation(x_bus))
            x_branch = self.dropout(self.activation(x_branch))
        return x_branch, x_bus


class GNN(torch.nn.Module):
    """Base class for node-only GNN baselines."""

    def __init__(self, args):
        super().__init__()
        self.num_layers = args.num_layers
        self.convs = torch.nn.ModuleList()
        self.dropout = Dropout(p=args.dropout)
        self.activation = ReLU(inplace=True)


class GCN(GNN):
    def __init__(self, args, in_channels, hidden_channels, out_channels):
        super().__init__(args)
        self.name = "GCN"
        for layer_id in range(self.num_layers):
            layer_in = in_channels if layer_id == 0 else hidden_channels
            layer_out = out_channels if layer_id == self.num_layers - 1 else hidden_channels
            self.convs.append(GCNConv(layer_in, layer_out))

    def forward(self, x, edge_index):
        for layer_id, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if layer_id != self.num_layers - 1:
                x = self.dropout(self.activation(x))
        return x


class SAGE(GNN):
    def __init__(self, args, in_channels, hidden_channels, out_channels):
        super().__init__(args)
        self.name = "GraphSAGE"
        for layer_id in range(self.num_layers):
            layer_in = in_channels if layer_id == 0 else hidden_channels
            layer_out = out_channels if layer_id == self.num_layers - 1 else hidden_channels
            self.convs.append(SAGEConv(layer_in, layer_out, normalize=False, root_weight=True))

    def forward(self, x, edge_index):
        edge_index = to_undirected(edge_index)
        for layer_id, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if layer_id != self.num_layers - 1:
                x = self.dropout(self.activation(x))
        return x


class GAT(GNN):
    def __init__(self, args, in_channels, hidden_channels, out_channels):
        super().__init__(args)
        self.name = "GAT"
        heads = 4
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, concat=True))
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads=1, concat=False))

    def forward(self, x, edge_index):
        for layer_id, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if layer_id != len(self.convs) - 1:
                x = self.dropout(self.activation(x))
        return x


class GATE(GNN):
    def __init__(self, args, in_channels, edge_dim, hidden_channels, out_channels):
        super().__init__(args)
        self.name = "GATE"
        heads = 4
        self.convs.append(GATConv(in_channels, hidden_channels, edge_dim=edge_dim, heads=heads, concat=True))
        self.convs.append(GATConv(heads * hidden_channels, out_channels, heads=1, concat=False))

    def forward(self, x, edge_index, edge_attr):
        for layer_id, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if layer_id != len(self.convs) - 1:
                x = self.dropout(self.activation(x))
        return x


class GIN(GNN):
    def __init__(self, args, in_channels, hidden_channels, out_channels):
        super().__init__(args)
        self.name = "GIN"
        current_channels = in_channels
        for _ in range(self.num_layers):
            mlp = MLP([current_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(mlp, train_eps=True))
            current_channels = hidden_channels
        self.mlp = MLP([hidden_channels, hidden_channels, out_channels], norm=None, dropout=0.0)

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = self.dropout(self.activation(conv(x, edge_index)))
        return self.mlp(x)


# Backward-compatible alias for earlier code using the old name.
NEA_GNN = PICE_GNN
