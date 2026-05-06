"""Post-processing utilities for PICE-GNN graph samples."""

from __future__ import annotations

import torch

from data import PairData
from utils import (
    build_line_graph_edge_index,
    build_node_edge_index,
    compute_dc_ptdf,
    init_line_graph_structural_encoding,
    init_structural_encoding_from_edges,
)

# Feature positions in edge_attr before structural encodings are appended.
BRANCH_R_FEATURE = 3
BRANCH_X_FEATURE = 4


def revise_data_index(
    data_list,
    node_rw_dim: int = 5,
    line_rw_dim: int = 6,
    slack_bus: int = 0,
):
    """Add structural encodings, line-graph indices, incidence indices, and PTDF.

    The raw samples generated from AC-CFM share the same grid topology. This
    function computes topology-dependent tensors once from the first sample and
    attaches them to every graph sample.
    """
    if data_list is None or len(data_list) == 0:
        raise ValueError("data_list must contain at least one graph sample.")

    first_graph = data_list[0]
    edge_index = first_graph.edge_index
    num_nodes = first_graph.x.size(0)
    reactance = first_graph.edge_attr[:, BRANCH_X_FEATURE]
    resistance = first_graph.edge_attr[:, BRANCH_R_FEATURE]

    line_graph_edge_index = build_line_graph_edge_index(edge_index)
    node_edge_index = build_node_edge_index(edge_index)

    node_structural_encoding = init_structural_encoding_from_edges(
        edge_index=edge_index,
        resistance=resistance,
        reactance=reactance,
        num_nodes=num_nodes,
        rw_dim=node_rw_dim,
    )

    branch_admittance = torch.abs(1.0 / (resistance + 1j * reactance + 1e-12)).float()
    branch_structural_encoding = init_line_graph_structural_encoding(
        line_graph_edge_index=line_graph_edge_index,
        branch_admittance=branch_admittance,
        rw_dim=line_rw_dim,
    )

    ptdf = compute_dc_ptdf(edge_index=edge_index, reactance=reactance, num_nodes=num_nodes, slack_bus=slack_bus)

    revised_data = []
    for graph in data_list:
        revised_graph = PairData(
            x=torch.cat([graph.x, node_structural_encoding], dim=1),
            edge_index=edge_index,
            edge_attr=torch.cat([graph.edge_attr, branch_structural_encoding], dim=1),
            y=graph.y,
            edge_label=graph.edge_label,
            line_graph_edge_index=line_graph_edge_index,
            node_edge_index_0=node_edge_index[0],
            node_edge_index_1=node_edge_index[1],
            ptdf=ptdf,
        )
        revised_data.append(revised_graph)

    return revised_data
