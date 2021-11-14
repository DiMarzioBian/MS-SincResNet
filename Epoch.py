import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

from xxMusic.Metrics import calc_voting_accuracy
from Utils import set_optimizer_lr


def train_epoch(model, data, opt, optimizer):
    """
    Flow for each epoch
    """
    num_data = data.dataset.length
    num_pred_correct_epoch = 0
    loss_epoch = 0

    model.train()

    print('\n- Learning rate is: ', str(optimizer.param_groups[0]['lr']))

    for batch in tqdm(data, desc='- (Training)   ', leave=False):
        wave, y_gt = map(lambda x: x.to(opt.device), batch)
        loss_batch, num_pred_correct_batch = model.loss(wave, y_gt)
        loss_batch.backward()
        optimizer.step()
        optimizer.zero_grad()

        num_pred_correct_epoch += num_pred_correct_batch
        loss_epoch += loss_batch * batch[1].shape[0]

    return loss_epoch / num_data, num_pred_correct_epoch / num_data


def test_epoch(model, data, gt_voting, opt, dataset):
    """
    Give prediction on test set
    """
    num_data = data.dataset.length
    num_pred_correct_epoch = 0
    loss_epoch = 0
    i = 0

    y_pred_epoch = (torch.ones(data.dataset.length) * (-1)).to(opt.device)
    model.eval()

    for batch in tqdm(data, desc='- (Testing)   ', leave=False):

        if opt.is_distributed:
            wave, y_gt_batch = map(lambda x: x.to(opt.local_rank), batch)
            loss_batch, num_pred_correct_batch, y_pred_batch = model.module.predict(wave, y_gt_batch)
        else:
            wave, y_gt_batch = map(lambda x: x.to(opt.device), batch)
            loss_batch, num_pred_correct_batch, y_pred_batch = model.predict(wave, y_gt_batch)

        num_pred_correct_epoch += num_pred_correct_batch
        loss_epoch += loss_batch * batch[1].shape[0]
        y_pred_epoch[i * data.batch_size: (i + 1) * data.batch_size] = y_pred_batch
        i += 1

    # Voting accuracy
    acc_voting = calc_voting_accuracy(y_pred_epoch, gt_voting, opt.sample_splits_per_track)

    return loss_epoch / num_data, num_pred_correct_epoch / num_data, acc_voting
