"""Training and evaluation loop for PICE-GNN experiments."""

from __future__ import annotations

import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch.optim import Adam, SGD

from Early_stopping import EarlyStopping
from f_loss import BinaryFocalLoss
from utils import accuracy2


class Trainer:
    def __init__(self, args, gnn=None, classifier=None, classifier_e=None, optimizer: str = "adam"):
        self.max_epochs = args.epochs
        self.device = args.device
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.gnn = gnn
        self.classifier = classifier
        self.classifier_e = classifier_e
        self.optimizer_name = optimizer
        self.beta = args.beta
        self.log_file = getattr(args, "log_file", "train.log")

        self.focal_loss = BinaryFocalLoss(alpha=0.25, gamma=2)
        self.early_stopping = EarlyStopping("./")

    def configure_optimizers(self):
        params = []
        for module in [self.gnn, self.classifier, self.classifier_e]:
            if module is not None:
                params.extend(param for param in module.parameters() if param.requires_grad)

        if self.optimizer_name == "sgd":
            return SGD(params, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "adam":
            return Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        print(f"Unsupported optimizer: {self.optimizer_name}")
        sys.exit(1)

    def forward(self, batch):
        if self.gnn.name in ["GCN", "GraphSAGE", "GIN", "GAT"]:
            embeds_v = self.gnn(batch.x, batch.edge_index)
            logits_v = self.classifier(embeds_v)
            logits_e = self._edge_logits_from_node_embeddings(embeds_v, batch.edge_index)
        elif self.gnn.name == "GATE":
            embeds_v = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
            logits_v = self.classifier(embeds_v)
            logits_e = self._edge_logits_from_node_embeddings(embeds_v, batch.edge_index)
        elif self.gnn.name == "PICE_GNN":
            batch.node_edge_index = torch.stack([batch.node_edge_index_0, batch.node_edge_index_1], dim=0)
            embeds_e, embeds_v = self.gnn(
                batch.x,
                batch.edge_attr,
                batch.edge_index,
                batch.node_edge_index,
                batch.line_graph_edge_index,
            )
            logits_v = self.classifier(embeds_v)
            logits_e = self.classifier_e(embeds_e)
        else:
            raise ValueError(f"Unsupported model name: {self.gnn.name}")

        return logits_e, logits_v

    @staticmethod
    def _edge_logits_from_node_embeddings(embeds_v: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        source_v = embeds_v[edge_index[0]]
        target_v = embeds_v[edge_index[1]]
        prob = torch.sigmoid((source_v * target_v).sum(dim=1, keepdim=True))
        return torch.cat([1.0 - prob, prob], dim=1)

    def _loss(self, logits_e, logits_v, batch):
        edge_loss = self.focal_loss(logits_e[:, 1].unsqueeze(1), batch.edge_label[:, 1].unsqueeze(1))
        node_loss = self.focal_loss(logits_v[:, 1].unsqueeze(1), batch.y[:, 1].unsqueeze(1))
        return edge_loss + self.beta * node_loss

    def train(self, dataloader) -> Tuple[float, float, float, float, float, float, float]:
        self.gnn.train()
        self.classifier.train()
        if self.classifier_e is not None:
            self.classifier_e.train()

        total_loss = 0.0
        edge_y_true, edge_y_pred, edge_scores = [], [], []

        for batch in dataloader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits_e, logits_v = self.forward(batch)
            loss = self._loss(logits_e, logits_v, batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            edge_y_true.extend(batch.edge_label.argmax(dim=1).detach().cpu().tolist())
            edge_y_pred.extend(logits_e.argmax(dim=1).detach().cpu().tolist())
            edge_scores.extend(logits_e[:, 1].detach().cpu().tolist())

        edge_acc, edge_pre, edge_rec, edge_f1, edge_bacc, edge_auc = accuracy2(
            edge_y_true, edge_y_pred, edge_scores
        )
        return (
            total_loss / max(len(dataloader), 1),
            edge_acc * 100,
            edge_pre * 100,
            edge_rec * 100,
            edge_f1 * 100,
            edge_bacc * 100,
            edge_auc * 100,
        )

    def fit(self, train_loader, val_loader, test_loader, verbose: bool = True) -> Dict[str, float]:
        self.gnn = self.gnn.to(self.device)
        self.classifier = self.classifier.to(self.device)
        if self.classifier_e is not None:
            self.classifier_e = self.classifier_e.to(self.device)

        self.optimizer = self.configure_optimizers()

        best_score = -float("inf")
        best_epoch = 0
        best_results = {}

        for epoch in range(1, self.max_epochs + 1):
            train_metrics = self.train(train_loader)
            val_metrics = self.test(val_loader)
            test_metrics = self.test(test_loader)

            current_score = self._selection_score(val_metrics)
            if current_score > best_score:
                best_score = current_score
                best_epoch = epoch
                best_results = self._format_results(test_metrics)
                best_results["best_epoch"] = best_epoch

            if verbose:
                self._print_epoch(epoch, train_metrics, val_metrics, test_metrics)

            self.early_stopping(current_score, self.gnn, self.classifier, self.classifier_e)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        if verbose and best_results:
            print("Best test results:")
            for key, value in best_results.items():
                print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        return best_results

    @torch.no_grad()
    def test(self, dataloader):
        self.gnn.eval()
        self.classifier.eval()
        if self.classifier_e is not None:
            self.classifier_e.eval()

        total_loss = 0.0
        edge_y_true, edge_y_pred, edge_scores = [], [], []
        node_y_true, node_y_pred, node_scores = [], [], []

        for batch in dataloader:
            batch = batch.to(self.device)
            logits_e, logits_v = self.forward(batch)
            loss = self._loss(logits_e, logits_v, batch)

            branch_in_service = batch.edge_attr[:, 0] > 0
            logits_e = logits_e.clone()
            logits_e[~branch_in_service] = batch.edge_label[~branch_in_service]

            total_loss += loss.item()
            edge_y_true.extend(batch.edge_label.argmax(dim=1).detach().cpu().tolist())
            edge_y_pred.extend(logits_e.argmax(dim=1).detach().cpu().tolist())
            edge_scores.extend(logits_e[:, 1].detach().cpu().tolist())
            node_y_true.extend(batch.y.argmax(dim=1).detach().cpu().tolist())
            node_y_pred.extend(logits_v.argmax(dim=1).detach().cpu().tolist())
            node_scores.extend(logits_v[:, 1].detach().cpu().tolist())

        edge_acc, edge_pre, edge_rec, edge_f1, edge_bacc, edge_auc = accuracy2(
            edge_y_true, edge_y_pred, edge_scores
        )
        node_acc, node_pre, node_rec, node_f1, node_bacc, node_auc = accuracy2(
            node_y_true, node_y_pred, node_scores
        )
        return (
            total_loss / max(len(dataloader), 1),
            edge_acc * 100,
            edge_pre * 100,
            edge_rec * 100,
            edge_f1 * 100,
            edge_bacc * 100,
            edge_auc * 100,
            node_acc * 100,
            node_pre * 100,
            node_rec * 100,
            node_f1 * 100,
            node_bacc * 100,
            node_auc * 100,
        )

    @staticmethod
    def _selection_score(metrics_tuple) -> float:
        return float(np.nanmean([metrics_tuple[1], metrics_tuple[2], metrics_tuple[3], metrics_tuple[4], metrics_tuple[5],
                                 metrics_tuple[7], metrics_tuple[8], metrics_tuple[9], metrics_tuple[10], metrics_tuple[11]]))

    @staticmethod
    def _format_results(metrics_tuple) -> Dict[str, float]:
        keys = [
            "loss",
            "edge_acc",
            "edge_precision",
            "edge_recall",
            "edge_f1",
            "edge_bacc",
            "edge_auc",
            "node_acc",
            "node_precision",
            "node_recall",
            "node_f1",
            "node_bacc",
            "node_auc",
        ]
        return dict(zip(keys, metrics_tuple))

    def _print_epoch(self, epoch: int, train_metrics, val_metrics, test_metrics):
        message = (
            f"Epoch [{epoch}/{self.max_epochs}] | "
            f"Train Loss {train_metrics[0]:.4f}, Edge Acc {train_metrics[1]:.2f}, "
            f"Edge F1 {train_metrics[4]:.2f}, Edge BAcc {train_metrics[5]:.2f} | "
            f"Val Loss {val_metrics[0]:.4f}, Edge F1 {val_metrics[4]:.2f}, "
            f"Node F1 {val_metrics[10]:.2f}, Mean Score {self._selection_score(val_metrics):.2f} | "
            f"Test Edge F1 {test_metrics[4]:.2f}, Test Node F1 {test_metrics[10]:.2f}"
        )
        print(message)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as file:
                file.write(message + "\n")
