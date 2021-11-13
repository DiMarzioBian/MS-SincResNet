from typing import Tuple
import argparse
import pickle
import os
import random
from scipy import signal

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

track_dict = pickle.load(open('_data/FMA_small/track_genre.pkl', 'rb'))


def load_FMA_small_item(filename: str, path: str, ext_audio: str = ".mp3") -> Tuple[torch.Tensor, int, str]:
    """
    Loads a file from the dataset and returns the raw waveform
    as a Torch Tensor, its sample rate as an integer, and its
    genre as a string.
    """
    # Read wav
    file_audio = os.path.join(path, filename)
    waveform, sample_rate = torchaudio.load(file_audio)

    return waveform, sample_rate, track_dict[filename]


class FMA_small_3s(Dataset):
    """ Event stream dataset. """

    def __init__(self,
                 list_filename: list,
                 new_sr: int,
                 sigma_gnoise: int = 0.01,
                 hop_gap: float = 0.5,
                 sample_splits_per_track: int = 100,
                 augment: bool = False,
                 root: str = '_data/FMA_small/fma_small',):
        """
        Instancelize FMA small, indexing clips by enlarged indices and map label to integers.
        """
        self._walker = list(track_dict.keys())

        self.root = root
        # self.old_sr = 22050
        self.new_sr = new_sr  # 16000
        self.sigma_gnoise = sigma_gnoise
        self.hop_gap = hop_gap
        self.sample_splits_per_track = sample_splits_per_track
        self.augment = augment

        self.total_num_splits = int(30.5 // (3+hop_gap))
        self._ext_audio = ".mp3"

        self.length = len(self._walker) * self.sample_splits_per_track

        self.table_random = [[]] * len(self._walker)
        for i in range(len(self._walker)):
            self.table_random[i] = random.sample(range(self.total_num_splits), self.sample_splits_per_track)

        file = open('_data/FMA_small/track_genre.pkl', 'rb')
        self.mapper_label = pickle.load(file)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Each returned element is a tuple[data(torch.tensor), label(int)]
        """
        index_full, index_table_split = divmod(index, self.sample_splits_per_track)
        index_split = self.table_random[index_full][index_table_split]

        wave, sr, genre_str = load_FMA_small_item(self._walker[index_full], self.root, ".mp3")
        wave = signal.resample(wave.mean(0).detach().numpy(), self.new_sr * 30)

        start = int((3+self.hop_gap) * index_split * self.new_sr)
        end = int(start + 3*self.new_sr)
        wave = torch.from_numpy(wave[start: end]).float().unsqueeze_(dim=0)

        # augmentation
        if self.augment:
            wave *= torch.rand(1) * 0.2 + 0.9
            wave += torch.randn(wave.size()) * self.sigma_gnoise
        return wave, genre_str

    def shuffle(self):
        for i in range(len(self._walker)):
            self.table_random[i] = random.sample(range(self.total_num_splits), self.sample_splits_per_track)


def get_FMA_small_dataloader(opt: argparse.Namespace, train_list: list, val_list: list):
    """ Load data and prepare dataloader. """

    # Calculate how many 3s clips could be extracted from a 30s track as available maximum of 3s clips
    # opt.sample_splits_per_track is the needed number of samples which must not be greater than maximum of 3s clips
    opt.total_num_splits = int(30.5 // (3 + opt.hop_gap))
    if opt.sample_splits_per_track > opt.total_num_splits:
        opt.sample_splits_per_track = opt.total_num_splits

    # Instancelize dataset
    train_data = FMA_small_3s(list_filename=train_list, new_sr=opt.sample_rate, sigma_gnoise=opt.sigma_gnoise,
                              hop_gap=opt.hop_gap, sample_splits_per_track=opt.sample_splits_per_track, augment=True)

    val_data = FMA_small_3s(list_filename=val_list, new_sr=opt.sample_rate, sigma_gnoise=opt.sigma_gnoise,
                            hop_gap=opt.hop_gap, sample_splits_per_track=opt.sample_splits_per_track, augment=False)

    if opt.is_distributed:
        # Instancelize sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)

        # Instancelize dataloader
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                  sampler=train_sampler)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    else:
        # Instancelize dataloader
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, val_loader


def get_FMA_small_labels(list_filename):
    """ Return filtered_all file genre """
    return torch.IntTensor(list(track_dict.values()))

