import numpy as np
import torch
import torch.nn as nn

def CrossEntropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    loss =  -np.sum(y_true * np.log(y_pred))/y_true.shape[0]
    return loss


class BiFocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction):
        super(BiFocalLoss).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        y_pred = torch.clip(y_pred, 1e-15, 1-1e-15)
        ce_loss = -y_true * torch.log(y_pred) - (1-y_true)*torch.log(1-y_pred)
        focalloss = self.alpha * (1-y_pred) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focalloss)
        elif self.reduction == 'sum':
            return torch.sum(focalloss)
        else:
            return focalloss



class MulFocalLoss(nn.Module):
    def __init__(self, alpha, gamma, reduction):
        super(MulFocalLoss).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_true, y_pred):
        y_pred = torch.clip(y_pred, 1e-15, 1-1e-15)
        ce_loss = -y_true * torch.log(y_pred)
        focalloss = self.alpha * (1-y_pred) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focalloss)
        elif self.reduction == 'sum':
            return torch.sum(focalloss)
        else:
            return focalloss

