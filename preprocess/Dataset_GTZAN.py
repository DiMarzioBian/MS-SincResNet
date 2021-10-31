import os
from typing import Tuple, Optional, Union
import random

from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.datasets.utils import (download_url,
                                       extract_archive,
                                       )
import argparse
from preprocess.Constants_GTZAN import *


URL = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
_CHECKSUMS = {"http://opihi.cs.uvic.ca/sound/genres.tar.gz": "5b3d6dddb579ab49814ab86dba69e7c7"}

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
                 subset: str,
                 new_sr: int,
                 sigma_gnoise: int = 0,
                 hop_gap: float = 0.5,
                 splits_per_track: int = 100,
                 root: str = '_data/GTZAN/',
                 download: bool = False,
                 url: str = URL):
        """
        Instancelize GTZAN, indexing clips by enlarged indices and map label to integers.
        """
        # self.walker = GTZAN(root='_data/GTZAN/', download=True, folder_in_archive='genres', subset=subset)

        self.subset = subset
        self.root = root
        self.old_sr = 22050
        self.new_sr = new_sr  # 16000
        self.sigma_gnoise = sigma_gnoise
        self.hop_gap = hop_gap
        self.splits_per_track = splits_per_track
        self.download = download
        self.url = url

        self.num_label = len(mapper_genre)
        self.total_num_splits = int(30.5 // (3+hop_gap))
        self._ext_audio = ".wav"

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, 'genres')

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(self.url, None)
                    download_url(self.url, self.root, hash_value=checksum, hash_type="md5")
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        else:
            if self.subset == "training":
                self._walker = filtered_train
            elif self.subset == "validation":
                self._walker = filtered_valid
            elif self.subset == "testing":
                self._walker = filtered_test

        self.length = len(self._walker) * self.splits_per_track

        self.table_random = [[]] * len(self._walker)
        for i in range(len(self._walker)):
            self.table_random[i] = random.sample(range(self.total_num_splits), self.splits_per_track)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Each returned element is a tuple[data(torch.tensor), label(int)]
        """
        index_full, index_table_split = divmod(index, self.splits_per_track)
        index_split = self.table_random[index_full][index_table_split]

        wave, sr, genre_str = load_gtzan_item(self._walker[index_full], self._path, self._ext_audio)
        wave = signal.resample(wave.squeeze(0).detach().numpy(), self.new_sr * 30)

        start = int((3+self.hop_gap) * index_split * self.new_sr)
        end = int(start + 3*self.new_sr)
        wave = torch.from_numpy(wave[start: end]).float().unsqueeze_(dim=0)

        # augmentation
        wave = torchaudio.transforms.Vol(gain=0.9, gain_type="amplitude")(wave)
        wave += torch.randn(wave.shape[-1]).unsqueeze(0) * self.sigma_gnoise

        return wave, mapper_genre[genre_str]

    def shuffle(self):
        for i in range(len(self._walker)):
            self.table_random[i] = random.sample(range(self.total_num_splits), self.splits_per_track)


def get_GTZAN_dataloader(opt: argparse.Namespace):
    """ Load data and prepare dataloader. """

    opt.total_num_splits = int(30.5 // (3 + opt.hop_gap))
    if opt.splits_per_track > opt.total_num_splits:
        opt.splits_per_track = opt.total_num_splits

    train_data = GTZAN_3s(subset='training', new_sr=opt.sample_rate, sigma_gnoise=opt.sigma_gnoise, hop_gap=opt.hop_gap,
                          splits_per_track=opt.splits_per_track)

    val_data = GTZAN_3s(subset='validation', new_sr=opt.sample_rate, sigma_gnoise=opt.sigma_gnoise, hop_gap=opt.hop_gap,
                        splits_per_track=opt.total_num_splits)

    test_data = GTZAN_3s(subset='testing', new_sr=opt.sample_rate, sigma_gnoise=opt.sigma_gnoise, hop_gap=opt.hop_gap,
                         splits_per_track=opt.total_num_splits)

    trainloader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)

    valloader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    testloader = DataLoader(test_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return trainloader, valloader, testloader


def get_GTZAN_labels():
    gt_train = torch.IntTensor([mapper_genre[s[:-6]] for s in filtered_train])
    gt_val = torch.IntTensor([mapper_genre[s[:-6]] for s in filtered_valid])
    gt_test = torch.IntTensor([mapper_genre[s[:-6]] for s in filtered_test])

    return gt_train, gt_val, gt_test
