import numpy as np
import cv2
import torch
import torch.nn as nn

def DiceLoss(preds, labels, eps=1e-5, activation='sigmoid'):
    n, c, h, w = preds.shape
    if activation is None or activation == 'none':
        activation_fn = lambda x:x
    elif activation == 'sigmoid':
        activation_fn = nn.Sigmoid()
    else:
        raise NotImplementedError("Activation function not defined in loss function")

    preds = activation_fn(preds)
    N = labels.size(0)
    preds = preds.view(N, -1)
    labels = labels.view(N, -1)

    tp = torch.sum(labels * preds, dim=1)
    fp = torch.sum(preds, dim=1) - tp
    fn = torch.sum(labels, dim=1) - tp
    loss = (2 * tp + eps) / (tp + fp + fn + eps)

    return loss.sum() / N

def DiceLoss_th(preds, labels, th):
    return -1