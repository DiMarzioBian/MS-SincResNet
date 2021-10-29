from xxMusic.Models import *

import os
import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from preprocess.Dataset import get_GTZAN_dataloader
from Epoch import train_epoch, test_epoch


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-version', type=str, default='0.0')
    parser.add_argument('-load_state', default=False)  # Provide state_dict path for testing or continue training
    parser.add_argument('-save_state', default=False)  # Saving best or latest model state_dict
    parser.add_argument('-test_only', default=False)  # Enable to skip training session
    parser.add_argument('-test_original', default=False)  # Test the original pretrained MS-SincResNet

    parser.add_argument('-data', default='GTZAN')
    parser.add_argument('-sample_rate', type=int, default=16000)

    parser.add_argument('-epoch', type=int, default=30)
    parser.add_argument('-num_workers', type=int, default=2)
    parser.add_argument('-batch_size', type=int, default=4)
    parser.add_argument('-lr', type=float, default=1e-4)
    parser.add_argument('-lr_patience', type=int, default=10)
    parser.add_argument('-es_patience', type=int, default=10)

    parser.add_argument('-resnet_pretrained', default=True)
    # parser.add_argument('-resnet_freeze', default=False)

    opt = parser.parse_args()
    opt.device = torch.device('cuda')
    opt.log = '_result/log/' + opt.version + time.strftime("-%b_%d_%H_%M", time.localtime()) + '.txt'

    create_data_folders()
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
    trainloader, valloader, testloader = get_GTZAN_dataloader(opt.sample_rate, opt.batch_size, opt.num_workers)

    # Load Music model
    model = xxMusic(opt)

    if opt.load_state:
        model.load_state_dict(torch.load(opt.load_state)['state_dict'])
    else:
        model.initialize_weights()

    model.to(opt.device)
    model.eval()

    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=(0.9, 0.999), eps=1e-05, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, int(opt.lr_patience), gamma=0.5)

    print('\n[Info] Model parameters:\n')
    for k, v in vars(opt).items():
        print('         %s: %s' % (k, v))

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n[Info] Number of parameters:{}'.format(num_params))

    # Run model
    if not opt.test_only:
        train(opt, model, trainloader, valloader, optimizer, scheduler)

    test(opt, model, testloader)

def create_data_folders():
    try:
        os.makedirs('_data/GTZAN')
    except:
        pass

def train(opt, model, trainloader, valloader, optimizer, scheduler):

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
        loss_train, acc_train = train_epoch(model, trainloader, opt, optimizer)
        end = time.time()
        scheduler.step()

        print('\n- (Training) Loss:{loss: 8.5f}, accuracy:{acc: 8.4f}, elapse:{elapse:3.4f} min'
              .format(loss=loss_train, acc=acc_train, elapse=(time.time() - start) / 60))

        """ Validating """
        with torch.no_grad():
            loss_val, acc_val, acc_val_voting = test_epoch(model, valloader, opt, dataset='val')

        print('\n- (Validating) Loss:{loss: 8.5f}, accuracy:{acc: 8.4f} and voting accuracy:{voting: 8.4f}'
              .format(loss=loss_val, acc=acc_val, voting=acc_val_voting))

        """ Logging """
        with open(opt.log, 'a') as f:
            f.write('{epoch}, {time: 8.4f}, {loss_train: 8.5f}, {acc_train: 8.4f}, {loss_val: 8.5f}, {acc_val: 8.4f}, '
                    '{acc_val_voting: 8.4f}\n'
                    .format(epoch=epoch, time=(end - start) / 60, loss_train=loss_train, acc_train=acc_train,
                            loss_val=loss_val, acc_val=acc_val, acc_val_voting=acc_val_voting), )

        """ Early stopping """
        if best_acc < acc_val:
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
                print("\n- [Info]Early stopping with best loss: {loss: 8.5f}, best accuracy: {acc: 8.4f} "
                      "and best voting accuracy: {voting: 8.4f}"
                      .format(acc=best_acc, loss=best_loss, voting=best_acc_voting), )
                break

    """ Reloading best model """
    print('\n- [Info] Final accuracy: {acc: 8.4f}, final accuracy: {loss: 8.5f}'
          .format(acc=best_acc, loss=best_loss), )

    with open(opt.log, 'a') as f:
        # Save hyperparameters
        for k, v in vars(opt).items():
            f.write('\n%s: %s' % (k, v))

    model.load_state_dict(model_best)
    if opt.save_state:
        torch.save(model_best, '_result/model/xxMusic-v' + opt.version + '-' +
                   time.strftime("-%b_%d_%H_%M", time.localtime()) + '.pth')


def test(opt, model, testloader):
    print('\n[ Epoch testing ]')

    with torch.no_grad():
        loss_test, acc_test, acc_test_voting = test_epoch(model, testloader, opt, dataset='test')

    print('\n- [Info] Test loss:{loss: 8.4f}, accuracy:{acc: 8.4f}, voting accuracy:{voting: 8.4f}'
          .format(acc=acc_test, loss=loss_test, voting=acc_test_voting), )

    with open(opt.log, 'a') as f:
        f.write('\nTest loss:{loss: 8.4f}, accuracy:{acc: 8.4f}, voting accuracy:{voting: 8.4f}'
                .format(acc=acc_test, loss=loss_test, voting=acc_test_voting), )

    print('\n------------------------ Finished. ------------------------\n')


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    seed = 0
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    main()
