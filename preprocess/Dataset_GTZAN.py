import os
from typing import Tuple
import random

from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import argparse
import numpy as np
from pyrubberband.pyrb import *


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


def load_gtzan_item(filename: str, path: str, ext_audio: str = ".wav") -> Tuple[torch.Tensor, int, str]:
    """
    Loads a file from the dataset and returns the raw waveform
    as a Torch Tensor, its sample rate as an integer, and its
    genre as a string.
    """
    # Filenames are of the form label.id, e.g. blues.00078
    label, _ = filename.split(".")

    # Read wav
    file_audio = os.path.join(path, label, filename + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return waveform, sample_rate, label


class GTZAN_3s(Dataset):
    """ Event stream dataset. """

    def __init__(self,
                 list_filename: list,
                 new_sr: int,
                 sigma_gnoise: int = 0,
                 hop_gap: float = 0.5,
                 sample_splits_per_track: int = 100,
                 augment: bool = False,
                 time_stretch_factor: float = 1.0,
                 augment_probability: float = 0.5,
                 pitch_shift_steps: float = 0.0,
                 root: str = '_data/GTZAN/',):
        """
        Instancelize GTZAN, indexing clips by enlarged indices and map label to integers.
        """
        self._walker = list_filename

        self.root = root
        self.old_sr = 22050
        self.new_sr = new_sr  # 16000
        self.sigma_gnoise = sigma_gnoise
        self.hop_gap = hop_gap

        #Time stretch augmentation trial
        self.time_stretch_factor = time_stretch_factor
        self.augment_probability = augment_probability

        #pitch shift augmentation
        self.pitch_shift_steps = pitch_shift_steps

        self.sample_splits_per_track = sample_splits_per_track
        self.augment = augment

        self.num_label = len(mapper_genre)
        self.total_num_splits = int(30.5 // (3+hop_gap))
        self._ext_audio = ".wav"

        self._path = os.path.join(root, 'genres')

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        self.length = len(self._walker) * self.sample_splits_per_track

        #random augment time stretch and pitch shift
        self.list_random_augment = random.sample(range(len(self._walker)),
                                                 int(np.floor(len(self._walker) * self.augment_probability)))

        self.table_random = [[]] * len(self._walker)
        for i in range(len(self._walker)):
            self.table_random[i] = random.sample(range(self.total_num_splits), self.sample_splits_per_track )

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Each returned element is a tuple[data(torch.tensor), label(int)]
        """
        index_full, index_table_split = divmod(index, self.sample_splits_per_track)

        index_split = self.table_random[index_full][index_table_split]

        wave, sr, genre_str = load_gtzan_item(self._walker[index_full], self._path, self._ext_audio)
        wave = signal.resample(wave.squeeze(0).detach().numpy(), self.new_sr * 30)

        start = int((3 + self.hop_gap) * index_split * self.new_sr)
        end = int(start + 3*self.new_sr)
        wave = wave[start: end]

        # pitch shift if augment is enabled and if pitch shift steps is not zero and current clip is in the random list
        if self.augment and self.pitch_shift_steps != 0 and index_full in self.list_random_augment:
            wave = pitch_shift(wave, self.new_sr, self.pitch_shift_steps)

        # time stretch if augment is enabled and factor is less than 1 and the current clip is in the random list
        if self.augment and self.time_stretch_factor < 1.0 and index_full in self.list_random_augment:
            wave = time_stretch(wave,self.new_sr,self.time_stretch_factor)
            wave = wave[:self.new_sr*3]  # take only the first 3 seconds of the time stretched clip

        wave = torch.from_numpy(wave).float().unsqueeze_(dim=0)

        # amplitude and gnoise augmentation for all training set data
        if self.augment:
            wave *= torch.rand(1) * 0.2 + 0.9
            wave += torch.randn(wave.size()) * self.sigma_gnoise

        return wave, mapper_genre[genre_str]

    def shuffle(self):
        self.list_random_augment = random.sample(range(len(self._walker)),
                                                 int(np.floor(len(self._walker) * self.augment_probability)))
        for i in range(len(self._walker)):
            self.table_random[i] = random.sample(range(self.total_num_splits), self.sample_splits_per_track)


def get_GTZAN_dataloader(opt: argparse.Namespace, train_list: list, val_list: list):
    """ Load data and prepare dataloader. """

    # Calculate how many 3s clips could be extracted from a 30s track as available maximum of 3s clips
    # opt.sample_splits_per_track is the needed number of samples which must not be greater than maximum of 3s clips
    opt.total_num_splits = int(30.5 // (3 + opt.hop_gap))
    if opt.sample_splits_per_track > opt.total_num_splits:
        opt.sample_splits_per_track = opt.total_num_splits

    # Instancelize dataset
    train_data = GTZAN_3s(list_filename=train_list, new_sr=opt.sample_rate, sigma_gnoise=opt.sigma_gnoise,
                          hop_gap=opt.hop_gap, sample_splits_per_track=opt.sample_splits_per_track,
                          time_stretch_factor=opt.time_stretch_factor, pitch_shift_steps=opt.pitch_shift_steps,
                          augment_probability=opt.augment_probability, augment=True)

    val_data = GTZAN_3s(list_filename=val_list, new_sr=opt.sample_rate, sigma_gnoise=opt.sigma_gnoise,
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


def get_GTZAN_labels(list_filename: list):
    """ Return filtered_all file genre """
    return torch.IntTensor([mapper_genre[s[:-6]] for s in list_filename])
