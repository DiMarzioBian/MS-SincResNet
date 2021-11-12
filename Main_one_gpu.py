from xxMusic.Models import *

import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchaudio.datasets.utils import download_url, extract_archive

from preprocess.Dataset import getter_dataloader, get_num_label
from Epoch import train_epoch, test_epoch
from Utils import set_optimizer_lr, data_downloader



def main():
    """
    Preparation
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--version', type=str, default='1.5.1')
    parser.add_argument('--note', type=str, default='Minor bug fixes, multiple GPU logging need update.')
    parser.add_argument('--enable_spp', type=bool, default=True)  # Enable SPP layer instead of ResNet fc layer directly

    parser.add_argument('--data', default='GTZAN')  # ranging from 0 to 9, integer
    parser.add_argument('--enable_data_filtered', default=True)  # Enable data filtering
    parser.add_argument('--download', default=True)  # Download dataset
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--hop_gap', type=float, default=0.5)  # time gap between each adjacent splits in a track
    parser.add_argument('--sample_splits_per_track', type=int, default=4)  # Random sampling splits instead of using all
    parser.add_argument('--sigma_gnoise', type=float, default=0.004)
    parser.add_argument('--smooth_label', type=float, default=0.3)

    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--l2_reg', type=float, default=1e-5)
    parser.add_argument('--es_patience', type=int, default=15)
    parser.add_argument('--gamma_steplr', type=float, default=np.sqrt(0.1))

    parser.add_argument('--resnet_pretrained', default=True)
    parser.add_argument('--is_distributed', type=bool, default=False)

    opt = parser.parse_args()
    opt.log = '_result/log/v' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.txt'
    opt.device = torch.device('cuda')

    # Download dataset
    if opt.download:
        data_downloader(opt.data)

    # Print hyperparameters and settings
    print('\n[Info] Model settings:\n')
    for k, v in vars(opt).items():
        print('         %s: %s' % (k, v))

    with open(opt.log, 'a') as f:
        # Save hyperparameters
        for k, v in vars(opt).items():
            f.write('%s: %s\n' % (k, v))

    """
    Start modeling
    """
    # Import data
    data_getter = getter_dataloader(opt)
    opt.num_label = get_num_label(opt.data)

    cv_acc = np.zeros(10)
    cv_loss = np.zeros(10)
    cv_acc_voting = np.zeros(10)

    """ Iterate 10 folds """
    for fold in range(10):
        print("\n------------------------ Start fold:{fold} ------------------------\n".format(fold=fold))
        with open(opt.log, 'a') as f:
            f.write("\n------------------------ Start fold:{fold} ------------------------\n".format(fold=fold), )
            f.write('\nEpoch, Time, loss_tr, acc_tr, loss_val, acc_val, acc_val_voting\n')

        # Load Music model
        model = xxMusic(opt).to(opt.device)
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9,
                              weight_decay=opt.l2_reg, nesterov=True)
        scheduler = optim.lr_scheduler.StepLR(optimizer, int(opt.lr_patience), gamma=opt.gamma_steplr)

        # Load data
        print('\n[Info] Loading data...')
        trainloader, valloader, val_gt_voting = data_getter.get(fold)

        # Define logging variants
        best_loss = 1e9
        best_acc = 0
        best_acc_voting = 0
        patience = 0
        model_best = None

        for epoch in range(1, opt.epoch + 1):
            print('\n[ Epoch {epoch}]'.format(epoch=epoch))

            # """ Training """
            start = time.time()

            trainloader.dataset.shuffle()
            loss_train, acc_train = train_epoch(model, trainloader, opt, optimizer)
            scheduler.step()

            end = time.time()

            print('\n- (Training) Loss:{loss: 8.5f}, accuracy:{acc: 8.4f}, elapse:{elapse:3.4f} min'
                  .format(loss=loss_train, acc=acc_train, elapse=(time.time() - start) / 60))

            """ Validating """
            with torch.no_grad():
                loss_val, acc_val, acc_val_voting = test_epoch(model, valloader, val_gt_voting, opt, dataset='val')

            print('\n- (Validating) Loss:{loss: 8.5f}, accuracy:{acc: 8.4f} and voting accuracy:{voting: 8.4f}'
                  .format(loss=loss_val, acc=acc_val, voting=acc_val_voting))

            """ Logging """
            with open(opt.log, 'a') as f:
                f.write('{epoch}, {time: 8.4f}, {loss_train: 8.5f}, {acc_train: 8.4f}, {loss_val: 8.5f}, '
                        '{acc_val: 8.4f}, {acc_val_voting: 8.4f}\n'
                        .format(epoch=epoch, time=(end - start) / 60, loss_train=loss_train, acc_train=acc_train,
                                loss_val=loss_val, acc_val=acc_val, acc_val_voting=acc_val_voting), )

            """ Early stopping """
            if (best_acc_voting < acc_val_voting) or ((best_acc_voting == acc_val_voting) & (best_loss >= loss_val)):
                best_acc = acc_val
                best_loss = loss_val
                best_acc_voting = acc_val_voting
                patience = 0
                print("\n- New best performance logged.")
            else:
                patience += 1
                print("\n- Early stopping patience counter {} of {}".format(patience, opt.es_patience))

                if patience == opt.es_patience:
                    print("\n[Info] Stop training")
                    break

        """ Logging """
        cv_loss[fold] = best_loss
        cv_acc[fold] = best_acc
        cv_acc_voting[fold] = best_acc_voting

        print("\n[Info] Training stopped with best loss: {loss: 8.5f}, best accuracy: {acc: 8.4f} "
              "and best voting accuracy: {voting: 8.4f}\n"
              .format(loss=best_loss, acc=best_acc, voting=best_acc_voting), )

        with open(opt.log, 'a') as f:
            f.write("\n[Info] Training stopped with best loss: {loss: 8.5f}, best accuracy: {acc: 8.4f} "
                    "and best voting accuracy: {voting: 8.4f}"
                    .format(loss=best_loss, acc=best_acc, voting=best_acc_voting), )

            f.write("\n------------------------ Finished fold:{fold} ------------------------\n"
                    .format(fold=fold), )
        print("\n------------------------ Finished fold:{fold} ------------------------\n".format(fold=fold))

    """ Final logging """
    with open(opt.log, 'a') as f:
        f.write("\n[Info] Average cross validation loss: {loss: 8.5f}, best accuracy: {acc: 8.4f} "
                "and best voting accuracy: {voting: 8.4f}\n"
                .format(loss=np.mean(cv_loss), acc=np.mean(cv_acc), voting=np.mean(cv_acc_voting)), )
    print('\n[Info] 10-fold average loss: {loss: 8.5f}, average accuracy: {acc: 8.4f}, '
          'average voting accuracy: {voting: 8.4f}\n'
          .format(loss=np.mean(cv_loss), acc=np.mean(cv_acc), voting=np.mean(cv_acc_voting)), )
    print('\n------------------------ Finished. ------------------------\n')

# def test(opt, model, data_getter):
#     print('\n[ Epoch testing ]')
#
#     with torch.no_grad():
#         loss_test, acc_test, acc_test_voting = test_epoch(model, valloader, val_gt_voting, opt, dataset='test')
#
#     print('\n- [Info] Test loss:{loss: 8.4f}, accuracy:{acc: 8.4f}, voting accuracy:{voting: 8.4f}'
#           .format(acc=acc_test, loss=loss_test, voting=acc_test_voting), )
#
#     with open(opt.log, 'a') as f:
#         f.write('\nTest loss:{loss: 8.4f}, accuracy:{acc: 8.4f}, voting accuracy:{voting: 8.4f}'
#                 .format(acc=acc_test, loss=loss_test, voting=acc_test_voting), )


if __name__ == '__main__':
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    main()
