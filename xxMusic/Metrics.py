import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess.Dataset import get_GTZAN_labels


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, num_label):
        super(LabelSmoothingLoss, self).__init__()
        assert 0.0 < label_smoothing <= 1.0

        self.eps = label_smoothing
        self.num_label = num_label

    def forward(self, y_pred, y_gt):
        """
        For cross entropy calculation
        pred_i (FloatTensor): (batch_size) x n_classes
        gt_i (LongTensor): batch_size
        """
        one_hot = F.one_hot(y_gt, num_classes=self.num_label).float()
        one_hot_non_tgt = (1 - one_hot)
        one_hot_smooth = one_hot * (1 - self.eps) + one_hot_non_tgt * self.eps / self.num_label
        log_prb = F.log_softmax(y_pred, dim=-1)
        loss = -(one_hot_smooth * log_prb).sum(dim=-1)

        return loss.sum()


def calc_voting_accuracy(y_pred: torch.Tensor, y_gt: torch.Tensor, sample_splits_per_track: int):
    """
    Vote out the final predicted label by 10 split 3s clips.
    """

    y_voting = torch.ones(y_gt.shape[0]).to(y_pred.device).int() * (-1)
    y_pred = y_pred.int()
    y_gt = y_gt.to(y_pred.device)

    for i in range(y_gt.shape[0]):
        y_tmp = y_pred[i * sample_splits_per_track: (i+1) * sample_splits_per_track]
        y_voting[i] = torch.bincount(y_tmp).argmax()

    return y_gt.eq(y_voting).sum() / y_gt.shape[0]
