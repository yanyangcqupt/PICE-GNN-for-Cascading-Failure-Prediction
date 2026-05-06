"""Utility functions for PICE-GNN experiments."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import scipy.io
from scipy import sparse as sp
import torch
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, roc_curve
from torch_geometric.utils import degree


def data_split(data: Sequence, train_ratio: float = 0.60, val_ratio: float = 0.20, seed: int | None = None):
    """Randomly split a sequence of graph samples into train/validation/test subsets."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be in (0, 1).")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be in [0, 1).")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be smaller than 1.")

    n_samples = len(data)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    perm = torch.randperm(n_samples, generator=generator).tolist()

    n_train = round(n_samples * train_ratio)
    n_val = round(n_samples * val_ratio)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return [data[i] for i in train_idx], [data[i] for i in val_idx], [data[i] for i in test_idx]


def get_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Return classification accuracy in percent."""
    pred = pred.argmax(dim=1) if pred.dim() > 1 else pred
    target = target.argmax(dim=1) if target.dim() > 1 else target
    return (pred == target).sum().item() / target.numel() * 100.0


# Adapted from common scipy.io.loadmat helper for nested MATLAB structs.
def loadmat(obj):
    """Load MATLAB .mat content while converting MATLAB structs to dictionaries."""

    def _check_keys(d):
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        d = {}
        for field_name in matobj._fieldnames:
            elem = matobj.__dict__[field_name]
            if isinstance(elem, scipy.io.matlab.mat_struct):
                d[field_name] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[field_name] = _tolist(elem)
            else:
                d[field_name] = elem
        return d

    def _tolist(ndarray):
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, scipy.io.matlab.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    if isinstance(obj, str):
        data = scipy.io.loadmat(obj, struct_as_record=False, squeeze_me=True)
        return _check_keys(data)
    if isinstance(obj, scipy.io.matlab.mat_struct):
        return _todict(obj)
    if isinstance(obj, np.ndarray):
        return _tolist(obj)
    raise TypeError(f"Unsupported object type for loadmat: {type(obj)!r}")


def build_line_graph_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """Construct the line-graph adjacency of an original graph edge list."""
    edge_pairs = [(tuple(edge), i) for i, edge in enumerate(edge_index.t().tolist())]
    line_graph_edges = set()

    for edge_i, idx_i in edge_pairs:
        for shared_node in edge_i:
            incident_edges = (edge_index[0] == shared_node) | (edge_index[1] == shared_node)
            for idx_j in torch.where(incident_edges)[0].tolist():
                if idx_i != idx_j:
                    line_graph_edges.add(tuple(sorted((idx_i, idx_j))))

    if not line_graph_edges:
        return torch.empty((2, 0), dtype=torch.long)

    line_graph_edge_index = torch.tensor(sorted(line_graph_edges), dtype=torch.long).t().contiguous()
    return line_graph_edge_index


def build_node_edge_index(edge_index: torch.Tensor) -> torch.Tensor:
    """Build bus-branch incidence indices used for cross-graph message passing."""
    n_edges = edge_index.size(1)
    node_indices = torch.cat([edge_index[0], edge_index[1]], dim=0)
    branch_indices = torch.arange(n_edges, dtype=torch.long).repeat(2)
    return torch.stack([node_indices, branch_indices], dim=0)


def _admittance_magnitude(resistance: torch.Tensor, reactance: torch.Tensor) -> np.ndarray:
    resistance_np = resistance.detach().cpu().numpy().astype(float)
    reactance_np = reactance.detach().cpu().numpy().astype(float)
    return np.abs(1.0 / (resistance_np + 1j * reactance_np + 1e-12))


def init_structural_encoding_from_edges(
    edge_index: torch.Tensor,
    resistance: torch.Tensor,
    reactance: torch.Tensor,
    num_nodes: int,
    rw_dim: int = 16,
) -> torch.Tensor:
    """Admittance-magnitude-guided random-walk structural encoding for buses."""
    admittance = _admittance_magnitude(resistance, reactance)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=float)

    row = edge_index[0].detach().cpu().numpy()
    col = edge_index[1].detach().cpu().numpy()
    for idx, (src, dst) in enumerate(zip(row, col)):
        adjacency[src, dst] = admittance[idx]
        adjacency[dst, src] = admittance[idx]

    degree_sum = adjacency.sum(axis=1)
    degree_sum[degree_sum == 0] = 1.0
    transition = adjacency @ sp.diags(1.0 / degree_sum)

    encodings = [torch.from_numpy(transition.diagonal()).float()]
    transition_power = transition
    for _ in range(rw_dim - 1):
        transition_power = transition_power @ transition
        encodings.append(torch.from_numpy(transition_power.diagonal()).float())

    return torch.stack(encodings, dim=-1)


def init_line_graph_structural_encoding(
    line_graph_edge_index: torch.Tensor,
    branch_admittance: torch.Tensor,
    rw_dim: int = 16,
) -> torch.Tensor:
    """Admittance-magnitude-guided random-walk structural encoding for branches."""
    num_edges = branch_admittance.numel()
    if line_graph_edge_index.numel() == 0:
        return torch.zeros((num_edges, rw_dim), dtype=torch.float32)

    branch_adm = branch_admittance.detach().cpu().numpy().astype(float)
    adjacency = np.zeros((num_edges, num_edges), dtype=float)
    src = line_graph_edge_index[0].detach().cpu().numpy()
    dst = line_graph_edge_index[1].detach().cpu().numpy()

    for i, j in zip(src, dst):
        weight = branch_adm[i] + branch_adm[j]
        adjacency[i, j] = weight
        adjacency[j, i] = weight

    degree_sum = adjacency.sum(axis=1)
    degree_sum[degree_sum == 0] = 1.0
    transition = adjacency @ sp.diags(1.0 / degree_sum)

    encodings = [torch.from_numpy(transition.diagonal()).float()]
    transition_power = transition
    for _ in range(rw_dim - 1):
        transition_power = transition_power @ transition
        encodings.append(torch.from_numpy(transition_power.diagonal()).float())

    return torch.stack(encodings, dim=-1)


def compute_dc_ptdf(edge_index: torch.Tensor, reactance: torch.Tensor, num_nodes: int, slack_bus: int = 0) -> torch.Tensor:
    """Compute a DC-PTDF matrix with shape [num_branches, num_buses]."""
    n_edges = edge_index.size(1)
    row = edge_index[0].detach().cpu().numpy().astype(int)
    col = edge_index[1].detach().cpu().numpy().astype(int)
    x = reactance.detach().cpu().numpy().astype(float)
    x = np.where(np.abs(x) < 1e-12, 1e-12, x)

    incidence = np.zeros((n_edges, num_nodes), dtype=float)
    susceptance = 1.0 / x
    for edge_id, (src, dst) in enumerate(zip(row, col)):
        incidence[edge_id, src] = 1.0
        incidence[edge_id, dst] = -1.0

    b_line = np.diag(susceptance) @ incidence
    b_bus = incidence.T @ np.diag(susceptance) @ incidence

    keep = [i for i in range(num_nodes) if i != slack_bus]
    b_bus_reduced = b_bus[np.ix_(keep, keep)]
    inv_reduced = np.linalg.pinv(b_bus_reduced)

    inv_full = np.zeros((num_nodes, num_nodes), dtype=float)
    inv_full[np.ix_(keep, keep)] = inv_reduced
    ptdf = b_line @ inv_full
    return torch.tensor(ptdf, dtype=torch.float32)


def accuracy1(labels, output):
    """Return accuracy, precision, recall, F1, and balanced accuracy."""
    output = np.asarray(output)
    labels = np.asarray(labels)
    acc = accuracy_score(labels, output)
    precision = metrics.precision_score(labels, output, zero_division=1, pos_label=0)
    recall = metrics.recall_score(labels, output, zero_division=1, pos_label=0)
    f1 = metrics.f1_score(labels, output, zero_division=1, pos_label=0)
    balanced_accuracy = metrics.balanced_accuracy_score(labels, output)
    return acc, precision, recall, f1, balanced_accuracy


def accuracy2(labels, output, scores):
    """Return accuracy, precision, recall, F1, balanced accuracy, and ROC-AUC."""
    output = np.asarray(output)
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    acc = accuracy_score(labels, output)
    precision = metrics.precision_score(labels, output, zero_division=1, pos_label=0)
    recall = metrics.recall_score(labels, output, zero_division=1, pos_label=0)
    f1 = metrics.f1_score(labels, output, zero_division=1, pos_label=0)
    balanced_accuracy = metrics.balanced_accuracy_score(labels, output)

    try:
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
    except ValueError:
        roc_auc = float("nan")

    return acc, precision, recall, f1, balanced_accuracy, roc_auc


# Backward-compatible aliases used by earlier scripts.
init_structural_encoding = init_structural_encoding_from_edges
init_structural_encoding_line_graph = init_line_graph_structural_encoding
init_structural_encoding_str = init_line_graph_structural_encoding
