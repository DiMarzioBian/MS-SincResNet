from preprocess.Dataset_GTZAN import *


def get_dataloader(dataset, opt):
    """ Choose dataset. """
    if dataset == 'GTZAN':
        trainloader, valloader, testloader = get_GTZAN_dataloader(opt)
        return trainloader, valloader, testloader

    elif dataset == 'BALLROOM':
        raise RuntimeError('Dataset ' + dataset + ' not loaded so far!')

    elif dataset == 'ISMIR2004':
        raise RuntimeError('Dataset ' + dataset + ' not loaded so far!')

    raise RuntimeError('Dataset '+dataset+' not found!')
