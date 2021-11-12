import os
from preprocess.Dataset_GTZAN import *
from preprocess.Dataset_EBallroom import *
from preprocess.Dataset_FMA_small import *
from sklearn.model_selection import StratifiedKFold

from preprocess.Constants import *


class getter_dataloader(object):
    """ Choose dataset. """

    def __init__(self, opt):
        self.opt = opt
        dataset = self.opt.data
        enable_data_filtered = self.opt.enable_data_filtered

        if dataset == 'GTZAN':
            if enable_data_filtered:
                # Load filtered file names
                self.data_filename = GTZAN_filtered
                self.data_labels = get_GTZAN_labels(GTZAN_filtered)
            else:
                # Create unfiltered file names
                label = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
                unfiltered_all = []
                for lbl in label:
                    for i in range(100):
                        unfiltered_all.append(lbl + '.000' + str(i).zfill(2))
                self.data_filename = unfiltered_all
                self.data_labels = get_GTZAN_labels(unfiltered_all)

            self.get_dataset_dataloader = get_GTZAN_dataloader
            self.get_labels = get_GTZAN_labels

        elif dataset == 'EBallroom':
            label = ['chacha', 'jive', 'quickstep', 'rumba', 'samba', 'tango', 'viennesewaltz', 'waltz', 'foxtrot']
            unfiltered_all = []
            data_labels = []
            for lbl in label:
                path_genre = os.path.join('_data', dataset, lbl)
                for f in os.listdir(path_genre):
                    unfiltered_all.append(f)
                    data_labels.append(lbl)

            self.data_labels = data_labels
            self.data_filename = unfiltered_all
            self.get_dataset_dataloader = get_EBallroom_dataloader
            self.get_labels = get_EBallroom_labels

        elif dataset == 'FMA_small':
            raise RuntimeError('Dataset ' + dataset + ' not loaded so far!')
        else:
            raise RuntimeError('Dataset ' + dataset + ' not found!')

        self.data_splitter = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

    def get(self, fold):
        """
        Return
        """
        assert 0 <= fold <= 9
        for i, (train_index, val_index) in enumerate(self.data_splitter.split(self.data_filename, self.data_labels)):
            if i != fold:
                continue
            else:
                train_loader, val_loader = self.get_dataset_dataloader(self.opt,
                                                                       [self.data_filename[i] for i in train_index],
                                                                       [self.data_filename[i] for i in val_index])
                val_gt_voting = self.get_labels([self.data_filename[i] for i in val_index])
                return train_loader, val_loader, val_gt_voting


def get_num_label(dataset: str):
    if dataset == 'GTZAN':
        return 10
    elif dataset == 'EBallroom':
        return 9
    elif dataset == 'FMA_small':
        return 8
    else:
        raise RuntimeError('Dataset ' + dataset + ' not found!')
