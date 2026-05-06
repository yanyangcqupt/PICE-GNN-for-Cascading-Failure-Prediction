"""Dataset conversion utilities for AC-CFM cascading-failure samples.

The code expects MATLAB files under::

    data/accfm/<case_name>/raw/

Typical file names are ``init_pf_case24.mat``, ``result_pf_case24*.mat``,
and ``result_case24*.mat``. The datasets used in the paper are not included
in this repository.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

from utils import build_line_graph_edge_index, build_node_edge_index, loadmat

# MATPOWER column indices, converted from 1-based MATLAB indexing to Python indexing.
BR_R = 3 - 1
BR_X = 4 - 1
RATE_A = 6 - 1
BR_STATUS = 11 - 1
PF = 14 - 1
QF = 15 - 1
PT = 16 - 1
QT = 17 - 1

BUS_TYPE = 2 - 1
VM = 8 - 1
VMAX = 12 - 1
VMIN = 13 - 1

PG = 2 - 1
QG = 3 - 1
PMAX = 9 - 1
QMAX = 4 - 1
QMIN = 5 - 1


class PairData(Data):
    """PyG data object with correct batching offsets for node-edge graphs."""

    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        edge_label=None,
        line_graph_edge_index=None,
        node_edge_index_0=None,
        node_edge_index_1=None,
        ptdf=None,
    ):
        super().__init__()
        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=0.0, neginf=0.0)
        if y is not None:
            self.y = y
        if edge_label is not None:
            self.edge_label = edge_label
        if line_graph_edge_index is not None:
            self.line_graph_edge_index = line_graph_edge_index
        if node_edge_index_0 is not None:
            self.node_edge_index_0 = node_edge_index_0
        if node_edge_index_1 is not None:
            self.node_edge_index_1 = node_edge_index_1
        if ptdf is not None:
            self.ptdf = ptdf

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return self.x.size(0)
        if key == "line_graph_edge_index":
            return self.edge_attr.size(0)
        if key == "node_edge_index_0":
            return self.x.size(0)
        if key == "node_edge_index_1":
            return self.edge_attr.size(0)
        if key == "ptdf":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


class ACCFMDataset(InMemoryDataset):
    """PyG in-memory dataset for AC-CFM-generated cascading-failure samples."""

    def __init__(self, root: str, name: str, transform=None, pre_transform=None):
        self.name = name
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, "raw")

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, "processed")

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        raise RuntimeError(
            "Raw AC-CFM datasets are not distributed with this repository. "
            f"Please place .mat files under {self.raw_dir!r}."
        )

    def process(self):
        profile_file = self._find_first("init_pf*.mat")
        cascade_file = self._find_first("result_case*.mat")
        power_flow_files = sorted(glob.glob(os.path.join(self.raw_dir, "result_pf*.mat")))
        if not power_flow_files:
            raise FileNotFoundError(f"No result_pf*.mat files found in {self.raw_dir!r}.")

        data_list: List[PairData] = []

        initial_scenarios = self._load_scenarios(profile_file)
        initial_network = _scenario_to_dict(initial_scenarios[0])
        initial_branch_labels = np.asarray(initial_network["branch"])[:, BR_STATUS]
        initial_bus_labels = np.asarray(initial_network["bus"])[:, BUS_TYPE]
        data_list.append(self._build_graph(initial_network, initial_branch_labels, initial_bus_labels))

        cascade_scenarios = self._load_scenarios(cascade_file)
        branch_labels = []
        bus_labels = []
        for scenario in cascade_scenarios:
            network = _scenario_to_dict(scenario)
            branch_labels.append(np.asarray(network["branch"])[:, BR_STATUS])
            bus_labels.append(np.asarray(network["bus"])[:, BUS_TYPE])

        power_flow_scenarios = []
        for file_path in power_flow_files:
            power_flow_scenarios.extend(self._load_scenarios(file_path))

        n_samples = min(len(power_flow_scenarios), len(branch_labels), len(bus_labels))
        for idx in range(n_samples):
            network = _scenario_to_dict(power_flow_scenarios[idx])
            data_list.append(self._build_graph(network, branch_labels[idx], bus_labels[idx]))

        torch.save(self.collate(data_list), self.processed_paths[0])

    def _find_first(self, pattern: str) -> str:
        matches = sorted(glob.glob(os.path.join(self.raw_dir, pattern)))
        if not matches:
            raise FileNotFoundError(f"No file matching {pattern!r} found in {self.raw_dir!r}.")
        return matches[0]

    @staticmethod
    def _load_scenarios(file_path: str) -> list:
        mat = loadmat(file_path)
        candidates = [value for key, value in mat.items() if not key.startswith("__")]
        for value in candidates:
            if isinstance(value, list):
                return value
            if isinstance(value, np.ndarray):
                return list(np.ravel(value))
        raise ValueError(f"No MATLAB scenario array found in {file_path!r}.")

    def _build_graph(self, network: Dict[str, np.ndarray], branch_labels, bus_labels) -> PairData:
        bus = np.asarray(network["bus"], dtype=float)
        gen = np.asarray(network["gen"], dtype=float)
        branch = np.asarray(network["branch"], dtype=float)

        load_pq = bus[:, 2:4]
        before_bus_type = bus[:, BUS_TYPE]
        voltage_ratio_min = bus[:, VM] / (bus[:, VMIN] + 1e-10)
        voltage_ratio_max = bus[:, VM] / (bus[:, VMAX] + 1e-10)
        bus_features = np.concatenate(
            [
                before_bus_type.reshape(-1, 1),
                voltage_ratio_min.reshape(-1, 1),
                voltage_ratio_max.reshape(-1, 1),
                bus[:, 7:9],
            ],
            axis=1,
        )

        gen_idx = (gen[:, 0] - 1).astype(int)
        gen_pq = gen[:, [PG, QG]]
        net_pq = np.zeros((bus_features.shape[0], gen_pq.shape[1]))
        for gen_row, bus_idx in enumerate(gen_idx):
            if 0 <= bus_idx < net_pq.shape[0]:
                net_pq[bus_idx] += gen_pq[gen_row]
        apparent_pq = net_pq[:, 0:2] - load_pq
        bus_features = np.concatenate([bus_features, apparent_pq], axis=1)

        bus_labels = np.asarray(bus_labels).copy()
        bus_labels[bus_labels == 4] = 0
        bus_labels[bus_labels > 0] = 1

        row = branch[:, 0].astype(int) - 1
        col = branch[:, 1].astype(int) - 1
        edge_index = torch.tensor(np.stack([row, col]), dtype=torch.long)

        branch_status = branch[:, BR_STATUS]
        rate_a = branch[:, RATE_A]
        rate_a = np.where(np.abs(rate_a) < 1e-10, 1e-10, rate_a)
        flow = torch.tensor(branch[:, [PF, QF, PT, QT]], dtype=torch.float64)

        branch_raw = torch.tensor(branch[:, [BR_R, BR_X, RATE_A, PF, QF, PT, QT]], dtype=torch.float64)
        branch_status_tensor = torch.tensor(branch_status, dtype=torch.float64)
        apparent_flow_from = torch.sqrt(flow[:, 0] ** 2 + flow[:, 1] ** 2).view(-1, 1)
        apparent_flow_to = torch.sqrt(flow[:, 2] ** 2 + flow[:, 3] ** 2).view(-1, 1)
        rate_a_tensor = torch.tensor(rate_a, dtype=torch.float64).view(-1, 1)

        branch_features = torch.cat(
            [
                branch_status_tensor.view(-1, 1).float(),
                (apparent_flow_from / rate_a_tensor).float(),
                (apparent_flow_to / rate_a_tensor).float(),
                branch_raw.float(),
            ],
            dim=1,
        )

        bus_tensor = torch.tensor(bus_features, dtype=torch.float32)
        bus_tensor = torch.nan_to_num(bus_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        branch_features = torch.nan_to_num(branch_features, nan=0.0, posinf=0.0, neginf=0.0)

        bus_tensor = _min_max_normalize(bus_tensor)
        branch_features = torch.cat(
            [branch_features[:, :1], _min_max_normalize(branch_features[:, 1:])],
            dim=1,
        )

        line_graph_edge_index = build_line_graph_edge_index(edge_index)
        node_edge_index = build_node_edge_index(edge_index)

        edge_y = F.one_hot(torch.tensor(branch_labels, dtype=torch.long), num_classes=2).float()
        node_y = F.one_hot(torch.tensor(bus_labels, dtype=torch.long), num_classes=2).float()

        return PairData(
            x=bus_tensor,
            edge_index=edge_index,
            edge_attr=branch_features,
            y=node_y,
            edge_label=edge_y,
            line_graph_edge_index=line_graph_edge_index,
            node_edge_index_0=node_edge_index[0],
            node_edge_index_1=node_edge_index[1],
        )

    def __repr__(self):
        return f"ACCFMDataset(name={self.name!r})"


def _scenario_to_dict(scenario: Any) -> Dict[str, np.ndarray]:
    if isinstance(scenario, dict):
        return {key: np.squeeze(value) if isinstance(value, np.ndarray) else value for key, value in scenario.items()}
    if hasattr(scenario, "_fieldnames"):
        return {
            field_name: np.squeeze(getattr(scenario, field_name))
            if isinstance(getattr(scenario, field_name), np.ndarray)
            else getattr(scenario, field_name)
            for field_name in scenario._fieldnames
        }
    raise TypeError(f"Unsupported scenario type: {type(scenario)!r}")


def _min_max_normalize(tensor: torch.Tensor) -> torch.Tensor:
    min_value = tensor.min(dim=0, keepdim=True)[0]
    max_value = tensor.max(dim=0, keepdim=True)[0]
    return (tensor - min_value) / (max_value - min_value + 1e-10)


def load_dataset(name: str, root: str = "data/accfm") -> ACCFMDataset:
    """Load an AC-CFM dataset by case name."""
    return ACCFMDataset(root=root, name=name)


# Backward-compatible alias.
ACCFM = ACCFMDataset
