import torch
import torch.utils.data
from preprocess.GTZAN_override import GTZAN, load_gtzan_item
from scipy import signal
import random
import numpy as np
from xxMusic.Constants import *

import torchaudio
mapper_genre = {
    "blues": 0,
    "classical": 1,
    "country": 2,
    "disco": 3,
    "hiphop": 4,
    "jazz": 5,
    "metal": 6,
    "pop": 7,
    "reggae": 8,
    "rock": 9,
}


class GTZAN_3s(torch.utils.data.Dataset):
    """ Event stream dataset. """

    def __init__(self, subset, new_sr):
        """
        Instancelize GTZAN, indexing clips by enlarged indices and map label to integers.
        """
        self.walker = GTZAN(root='_data/GTZAN/', download=True, folder_in_archive='genres', subset=subset)
        self.k_splits = 10
        self.old_sr = 22050
        self.new_sr = new_sr  # 16000
        self.length = len(self.walker) * self.k_splits


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Each returned element is a tuple[data(torch.tensor), label(int)]
        """

        index_30, index_3 = divmod(index, self.k_splits)
        wave, sr, genre_str = self.walker[index_30]
        wave = signal.resample(wave.squeeze(0).detach().numpy(), self.new_sr * 30)
        wave = wave[3 * index_3 * self.new_sr: 3 * (index_3 + 1) * self.new_sr]
        wave = torch.from_numpy(wave).float().unsqueeze_(dim=0)
        #augmentation
        wave = torchaudio.transforms.Vol(gain=0.9, gain_type="amplitude")(wave)
        wave = wave + 0.02 * np.random.normal(len(wave))

        return wave, mapper_genre[genre_str]


def get_GTZAN_dataloader(sample_rate, batch_size, num_workers):
    """ Load data and prepare dataloader. """

    train_data = GTZAN_3s(subset='training', new_sr=sample_rate)
    val_data = GTZAN_3s(subset='validation', new_sr=sample_rate)

    test_data = GTZAN_3s(subset='testing', new_sr=sample_rate)

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    valloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return trainloader, valloader, testloader


def get_GTZAN_labels():
    gt_train = torch.IntTensor([mapper_genre[s[:-6]] for s in filtered_train])
    gt_val = torch.IntTensor([mapper_genre[s[:-6]] for s in filtered_valid])
    gt_test = torch.IntTensor([mapper_genre[s[:-6]] for s in filtered_test])

    return gt_train, gt_val, gt_test
