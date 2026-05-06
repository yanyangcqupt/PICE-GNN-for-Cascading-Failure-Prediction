import torch
import torch.nn as nn


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-8

    def forward(self, preds, labels):

        preds = preds.view(-1).float()
        labels = labels.view(-1).float()


        if torch.min(preds) < 0 or torch.max(preds) > 1:
            preds = torch.sigmoid(preds)

        log_p = torch.log(torch.clamp(preds, self.eps, 1 - self.eps))
        log_1_p = torch.log(torch.clamp(1 - preds, self.eps, 1 - self.eps))

        loss_1 = -self.alpha * (1 - preds) ** self.gamma * log_p * labels
        loss_0 = -(1 - self.alpha) * preds ** self.gamma * log_1_p * (1 - labels)

        return (loss_0 + loss_1).mean()