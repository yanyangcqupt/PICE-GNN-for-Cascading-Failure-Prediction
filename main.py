"""Train and evaluate PICE-GNN for cascading-failure prediction."""

from __future__ import annotations

import argparse
import random
import sys

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from data import load_dataset
from model import Classification, GAT, GATE, GCN, GIN, PICE_GNN, SAGE
from revise_data import revise_data_index
from train import Trainer
from utils import data_split


def get_arguments():
    parser = argparse.ArgumentParser(description="PICE-GNN for cascading failure prediction")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda", action="store_true", default=True, help="use CUDA when available")
    parser.add_argument("--gpu_id", type=int, default=0)

    parser.add_argument("--dataset", type=str, default="case39", help="dataset name, e.g., case39 or case118")
    parser.add_argument("--data_root", type=str, default="data/accfm", help="root folder of AC-CFM datasets")
    parser.add_argument("--train_ratio", type=float, default=0.60)
    parser.add_argument("--val_ratio", type=float, default=0.20)

    parser.add_argument(
        "--model",
        type=str,
        default="pice_gnn",
        choices=["pice_gnn", "nea_gnn", "gcn", "sage", "gat", "gate", "gin"],
        help="GNN architecture",
    )
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=16, help="hidden dimension of GNN embeddings")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--classifier_hidden", type=int, default=128)

    parser.add_argument("--beta", type=float, default=1.0, help="weight for bus/node loss")
    parser.add_argument("--class_weight_e", type=float, default=0.1, help="branch majority-class weight")
    parser.add_argument("--class_weight_n", type=float, default=0.1, help="bus majority-class weight")
    parser.add_argument("--log_file", type=str, default="train.log")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_arguments(args):
    print("Arguments:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")


def build_model(args, num_node_features: int, num_edge_features: int, ptdf: torch.Tensor):
    model_name = args.model.lower()
    if model_name == "nea_gnn":
        model_name = "pice_gnn"

    hidden_node = args.hidden
    hidden_edge = args.hidden
    edge_classifier = None

    if model_name == "gcn":
        gnn = GCN(args, num_node_features, args.hidden, args.hidden)
    elif model_name == "sage":
        gnn = SAGE(args, num_node_features, args.hidden, args.hidden)
    elif model_name == "gat":
        gnn = GAT(args, num_node_features, args.hidden, args.hidden)
    elif model_name == "gate":
        gnn = GATE(args, num_node_features, num_edge_features, args.hidden, args.hidden)
    elif model_name == "gin":
        gnn = GIN(args, num_node_features, args.hidden, args.hidden)
    elif model_name == "pice_gnn":
        add_feature = 1
        gnn = PICE_GNN(
            num_layers=args.num_layers,
            dropout=args.dropout,
            in_channels=[num_node_features, num_edge_features],
            hidden_channels=[args.hidden, add_feature, args.hidden, add_feature],
            out_channels=[args.hidden, add_feature, args.hidden, add_feature],
            ptdf=ptdf,
        )
        if args.num_layers % 2 == 0:
            hidden_node = args.hidden
            hidden_edge = args.hidden + add_feature
        else:
            hidden_node = args.hidden + add_feature
            hidden_edge = args.hidden
        edge_classifier = Classification(2, hidden_edge, args.classifier_hidden, 2)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    node_classifier = Classification(2, hidden_node, args.classifier_hidden, 2)
    return gnn, node_classifier, edge_classifier


def count_trainable_parameters(*modules) -> int:
    return sum(p.numel() for module in modules if module is not None for p in module.parameters() if p.requires_grad)


if __name__ == "__main__":
    args = get_arguments()
    set_seed(args.seed)
    print_arguments(args)

    args.loss_weight_e = torch.tensor([1.0, args.class_weight_e])
    args.loss_weight_n = torch.tensor([1.0, args.class_weight_n])

    raw_dataset = list(load_dataset(args.dataset, root=args.data_root))
    dataset = revise_data_index(raw_dataset)
    initial_profile = dataset[0]

    train_dataset, val_dataset, test_dataset = data_split(
        dataset[1:], train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    if args.cuda and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        args.device = torch.device(f"cuda:{args.gpu_id}")
        print(f"Using GPU: {torch.cuda.current_device()}")
    else:
        args.device = torch.device("cpu")
        print("Using CPU")

    num_node_features = initial_profile.x.shape[1]
    num_edge_features = initial_profile.edge_attr.shape[1]
    ptdf = initial_profile.ptdf

    gnn, node_classifier, edge_classifier = build_model(args, num_node_features, num_edge_features, ptdf)
    total_params = count_trainable_parameters(gnn, node_classifier, edge_classifier)
    print(f"Trainable parameters: {total_params:,} ({total_params / 1e6:.4f}M)")

    trainer = Trainer(args, gnn, node_classifier, edge_classifier)
    results = trainer.fit(train_loader, val_loader, test_loader)
    print("Best results:")
    print(results)
