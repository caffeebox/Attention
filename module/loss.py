import torch.nn as nn
import torch

def my_criterion():
    pass

def binary_focal_loss(alpha=0.25, gamma=2):
    def focal(y_pred, y_true):
        p_true = torch.where(y_true==1, y_pred, torch.ones_like(y_pred, dtype=torch.float))
        p_false = torch.where(y_true==0, y_pred, torch.zeros_like(y_true, dtype=torch.float))
        result_ture = -torch.sum(alpha * ((1-p_true)**gamma) * (p_true.log()))
        result_false = -torch.sum((1-alpha) * (p_false**gamma) * ((1-p_false).log()))
        return result_false + result_ture
    return focal

def bce_loss():
    return nn.BCELoss()


if __name__ == '__main__':
    y_ture = torch.tensor([[1, 0], [1, 0]])
    y_pred = torch.tensor([[0.2, 0.2], [0.8, 0.8]])
    criterion = binary_focal_loss()
    loss = criterion(y_pred, y_ture)
    print(loss.item())
