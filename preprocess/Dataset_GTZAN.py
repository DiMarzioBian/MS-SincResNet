import os
from typing import Tuple, Optional, Union

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
        self.download = download
        self.url = url

        self.num_label = len(mapper_genre)
        self.k_splits = 10
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
        self.length = len(self._walker) * self.k_splits

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Each returned element is a tuple[data(torch.tensor), label(int)]
        """
        index_30, index_3 = divmod(index, self.k_splits)
        wave, sr, genre_str = load_gtzan_item(self._walker[index_30], self._path, self._ext_audio)
        wave = signal.resample(wave.squeeze(0).detach().numpy(), self.new_sr * 30)
        wave = wave[3 * index_3 * self.new_sr: 3 * (index_3+1) * self.new_sr]
        wave = torch.from_numpy(wave).float().unsqueeze_(dim=0)

        # augmentation
        wave = torchaudio.transforms.Vol(gain=0.9, gain_type="amplitude")(wave)
        wave += torch.randn(wave.shape[-1]).unsqueeze(0) * self.sigma_gnoise

        return wave, mapper_genre[genre_str]


def get_GTZAN_dataloader(opt: argparse.Namespace):
    """ Load data and prepare dataloader. """
    new_sr = opt.sample_rate
    sigma_gnoise = opt.sigma_gnoise
    batch_size = opt.batch_size
    num_workers = opt.num_workers

    train_data = GTZAN_3s(subset='training', new_sr=new_sr, sigma_gnoise=sigma_gnoise)

    val_data = GTZAN_3s(subset='validation', new_sr=new_sr, sigma_gnoise=sigma_gnoise)

    test_data = GTZAN_3s(subset='testing', new_sr=new_sr, sigma_gnoise=sigma_gnoise)

    trainloader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    valloader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    testloader = DataLoader(test_data, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return trainloader, valloader, testloader


def get_GTZAN_labels():
    gt_train = torch.IntTensor([mapper_genre[s[:-6]] for s in filtered_train])
    gt_val = torch.IntTensor([mapper_genre[s[:-6]] for s in filtered_valid])
    gt_test = torch.IntTensor([mapper_genre[s[:-6]] for s in filtered_test])

    return gt_train, gt_val, gt_test
