from typing import Tuple
import os
import random
from scipy import signal
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import argparse


mapper_genre = {
    "chacha": 0,
    "jive": 1,
    "quickstep": 2,
    "rumba": 3,
    "samba": 4,
    "tango": 5,
    "viennesewaltz": 6,
    "waltz": 7,
    "foxtrot": 8,
}


def load_EBallroom_item(filename: str, path: str, ext_audio: str = ".mp3") -> Tuple[torch.Tensor, int, str]:
    """
    Loads a file from the dataset and returns the raw waveform
    as a Torch Tensor, its sample rate as an integer, and its
    genre as a string.
    """
    # Filenames are of the form label.id, e.g. blues.00078
    label, *_ = filename.split(".")

    # Read wav
    file_audio = os.path.join(path, label, filename[:-4] + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)

    return waveform, sample_rate, label


class EBallroom_3s(Dataset):
    """ Event stream dataset. """

    def __init__(self,
                 list_filename: list,
                 new_sr: int,
                 sigma_gnoise: int = 0.01,
                 hop_gap: float = 0.5,
                 sample_splits_per_track: int = 100,
                 augment: bool = False,
                 root: str = '_data/EBallroom/',):
        """
        Instancelize GTZAN, indexing clips by enlarged indices and map label to integers.
        """
        self._walker = list_filename

        self.root = root
        # self.old_sr = 22050
        self.new_sr = new_sr  # 16000
        self.sigma_gnoise = sigma_gnoise
        self.hop_gap = hop_gap
        self.sample_splits_per_track = sample_splits_per_track
        self.augment = augment

        self.num_label = len(mapper_genre)
        self.total_num_splits = int(30.5 // (3+hop_gap))
        self._ext_audio = ".mp3"

        # Discard cateogries: , 'Pasodoble', 'Salsa', Slowwaltz', 'Wcswing' to keep datset updpeedm
        self.label = ['chacha', 'jive', 'quickstep', 'rumba', 'samba', 'tango', 'viennesewaltz', 'waltz', 'foxtrot']

        self.length = len(self._walker) * self.sample_splits_per_track

        self.table_random = [[]] * len(self._walker)
        for i in range(len(self._walker)):
            self.table_random[i] = random.sample(range(self.total_num_splits), self.sample_splits_per_track)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Each returned element is a tuple[data(torch.tensor), label(int)]
        """
        index_full, index_table_split = divmod(index, self.sample_splits_per_track)
        index_split = self.table_random[index_full][index_table_split]

        wave, sr, genre_str = load_EBallroom_item(self._walker[index_full], self.root, ".mp3")
        wave = signal.resample(wave.mean(0).detach().numpy(), self.new_sr * 30)

        start = int((3+self.hop_gap) * index_split * self.new_sr)
        end = int(start + 3*self.new_sr)
        wave = torch.from_numpy(wave[start: end]).float().unsqueeze_(dim=0)

        # augmentation
        if self.augment:
            wave *= torch.rand(1) * 0.2 + 0.9
            wave += torch.randn(wave.size()) * self.sigma_gnoise
        return wave, mapper_genre[genre_str]

    def shuffle(self):
        for i in range(len(self._walker)):
            self.table_random[i] = random.sample(range(self.total_num_splits), self.sample_splits_per_track)


def get_EBallroom_dataloader(opt: argparse.Namespace, train_list: list, val_list: list):
    """ Load data and prepare dataloader. """

    # Calculate how many 3s clips could be extracted from a 30s track as available maximum of 3s clips
    # opt.sample_splits_per_track is the needed number of samples which must not be greater than maximum of 3s clips
    opt.total_num_splits = int(30.5 // (3 + opt.hop_gap))
    if opt.sample_splits_per_track > opt.total_num_splits:
        opt.sample_splits_per_track = opt.total_num_splits

    # Instancelize dataset
    train_data = EBallroom_3s(list_filename=train_list, new_sr=opt.sample_rate, sigma_gnoise=opt.sigma_gnoise,
                              hop_gap=opt.hop_gap, sample_splits_per_track=opt.sample_splits_per_track, augment=True)

    val_data = EBallroom_3s(list_filename=val_list, new_sr=opt.sample_rate, sigma_gnoise=opt.sigma_gnoise,
                            hop_gap=opt.hop_gap, sample_splits_per_track=opt.sample_splits_per_track, augment=False)

    if opt.is_distributed:
        # Instancelize sampler
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_data, shuffle=False)

        # Instancelize dataloader
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                  sampler=train_sampler)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                sampler=val_sampler)

    else:
        # Instancelize dataloader
        train_loader = DataLoader(train_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=False)

    return train_loader, val_loader


def get_EBallroom_labels(list_filename):
    """ Return filtered_all file genre """
    return torch.IntTensor([mapper_genre[s[:-11]] for s in list_filename])

