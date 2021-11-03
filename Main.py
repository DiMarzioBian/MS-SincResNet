from xxMusic.Models import *

import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from preprocess.Dataset import getter_dataloader
from Epoch import train_epoch, test_epoch
from Utils import adjust_learning_rate


def main(fold: int):
    parser = argparse.ArgumentParser()

    parser.add_argument('-version', type=str, default='1.0')
    parser.add_argument('-load_state', default=False)  # Provide state_dict path for testing or continue training
    parser.add_argument('-save_state', default=False)  # Saving best or latest model state_dict
    parser.add_argument('-test_only', default=False)  # Enable to skip training session
    parser.add_argument('-test_original', default=False)  # Deprecated, model structure has changed

    parser.add_argument('-data', default='GTZAN') # ranging from 0 to 9, integer
    parser.add_argument('-sample_rate', type=int, default=16000)
    parser.add_argument('-hop_gap', type=float, default=0.5)  # time gap between each adjacent splits in a track
    parser.add_argument('-splits_per_track', type=int, default=4)  # Random sample some splits instead of using all
    parser.add_argument('-sigma_gnoise', type=float, default=0.004)
    parser.add_argument('-smooth_label', type=float, default=0.3)

    parser.add_argument('-epoch', type=int, default=1)
    parser.add_argument('-num_workers', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=2)
    parser.add_argument('-manual_lr', default=False)
    parser.add_argument('-lr', type=float, default=1e-4)  # Enable manual_lr will override this lr
    parser.add_argument('-lr_patience', type=int, default=10)
    parser.add_argument('-es_patience', type=int, default=15)

    parser.add_argument('-resnet_pretrained', default=True)
    # parser.add_argument('-resnet_freeze', default=False)

    opt = parser.parse_args()
    opt.fold = fold
    opt.device = torch.device('cuda')
    opt.log = '_result/log/v'+opt.version+'-fold'+str(opt.fold)+time.strftime("-%b_%d_%H_%M", time.localtime())+'.txt'

    with open(opt.log, 'w') as f:
        f.write('Epoch, Time, loss_tr, acc_tr, loss_val, acc_val, acc_val_voting\n')

    # Test the original pretrained MS-SincResNet
    if opt.test_original:
        opt.save_state = False
        opt.test_only = True
        opt.load_state = "_trained/MS-SincResNet.tar"
        opt.data = 'GTZAN'

    # Import data
    print('\n[Info] Loading data...')
    data_getter = getter_dataloader(opt)
    trainloader, valloader, val_gt_voting = data_getter.get(opt.fold)
    opt.num_label = trainloader.dataset.num_label

    # Load Music model
    model = xxMusic(opt)

    if opt.load_state:
        model.load_state_dict(torch.load(opt.load_state)['state_dict'])
    else:
        model.initialize_weights()

    model.to(opt.device)

    optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=opt.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, int(opt.lr_patience), gamma=0.707)

    print('\n[Info] Model parameters:\n')
    for k, v in vars(opt).items():
        print('         %s: %s' % (k, v))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n[Info] Number of parameters:{}'.format(num_params))

    # Run model
    if not opt.test_only:
        # Train model
        train(opt, model, trainloader, valloader, val_gt_voting, optimizer, scheduler)
    else:
        # Test only
        if opt.load_state:
            model.load_state_dict(opt.load_state)
        test(opt, model, valloader, val_gt_voting)


def train(opt, model, trainloader, valloader, val_gt_voting, optimizer, scheduler):

    # Define logging variants
    best_loss = 1e9
    best_acc = 0
    best_acc_voting = 0
    patience = 0
    model_best = None

    for epoch in range(1, opt.epoch + 1):
        print('\n[ Epoch {epoch}]'.format(epoch=epoch))

        """ Training """
        start = time.time()

        if opt.manual_lr:
            adjust_learning_rate(optimizer, epoch)

        loss_train, acc_train = train_epoch(model, trainloader, opt, optimizer)
        end = time.time()
        trainloader.dataset.shuffle()

        if not opt.manual_lr:
            scheduler.step()

        print('\n- (Training) Loss:{loss: 8.5f}, accuracy:{acc: 8.4f}, elapse:{elapse:3.4f} min'
              .format(loss=loss_train, acc=acc_train, elapse=(time.time() - start) / 60))

        """ Validating """
        with torch.no_grad():
            loss_val, acc_val, acc_val_voting = test_epoch(model, valloader, val_gt_voting, opt, dataset='val')

        print('\n- (Validating) Loss:{loss: 8.5f}, accuracy:{acc: 8.4f} and voting accuracy:{voting: 8.4f}'
              .format(loss=loss_val, acc=acc_val, voting=acc_val_voting))

        """ Logging """
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {time: 8.4f}, {loss_train: 8.5f}, {acc_train: 8.4f}, {loss_val: 8.5f}, {acc_val: 8.4f}, '
                    '{acc_val_voting: 8.4f}\n'
                    .format(epoch=epoch, time=(end - start) / 60, loss_train=loss_train, acc_train=acc_train,
                            loss_val=loss_val, acc_val=acc_val, acc_val_voting=acc_val_voting), )

        """ Early stopping """
        if best_acc_voting < acc_val_voting:
            best_acc = acc_val
            best_loss = loss_val
            best_acc_voting = acc_val_voting
            patience = 0
            model_best = model.state_dict().copy()
            print("\n- New best performance logged.")
        else:
            patience += 1
            print("\n- Early stopping patience counter {} of {}".format(patience, opt.es_patience))

            if patience == opt.es_patience:

                print("\n[Info]Early stopping with best loss: {loss: 8.5f}, best accuracy: {acc: 8.4f} "
                      "and best voting accuracy: {voting: 8.4f}"
                      .format(acc=best_acc, loss=best_loss, voting=best_acc_voting), )

                with open(opt.log, 'a') as f:
                    f.write("\n[Info]Early stopping with best loss: {loss: 8.5f}, best accuracy: {acc: 8.4f} "
                            "and best voting accuracy: {voting: 8.4f}"
                            .format(acc=best_acc, loss=best_loss, voting=best_acc_voting), )
                break

    """ Reloading best model """
    model.load_state_dict(model_best)
    with open(opt.log, 'a') as f:
        # Save hyperparameters
        for k, v in vars(opt).items():
            f.write('\n%s: %s' % (k, v))

    model.load_state_dict(model_best)
    if opt.save_state:
        torch.save(model_best, '_result/model/xxMusic-v' + opt.version + '-' +
                   time.strftime("-%b_%d_%H_%M", time.localtime()) + '.pth')


def test(opt, model, valloader, val_gt_voting):
    print('\n[ Epoch testing ]')

    with torch.no_grad():
        loss_test, acc_test, acc_test_voting = test_epoch(model, valloader, val_gt_voting, opt, dataset='test')

    print('\n- [Info] Test loss:{loss: 8.4f}, accuracy:{acc: 8.4f}, voting accuracy:{voting: 8.4f}'
          .format(acc=acc_test, loss=loss_test, voting=acc_test_voting), )

    with open(opt.log, 'a') as f:
        f.write('\nTest loss:{loss: 8.4f}, accuracy:{acc: 8.4f}, voting accuracy:{voting: 8.4f}'
                .format(acc=acc_test, loss=loss_test, voting=acc_test_voting), )


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    for fold in range(10):
        main(2)

    print('\n------------------------ Finished. ------------------------\n')