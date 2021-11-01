import os
from pathlib import Path
from typing import Tuple, Optional, Union

import torchaudio
from torch import Tensor
from torch.utils.data import Dataset
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
)
import librosa
# Load own data splits instead of the preset
from xxMusic.Constants_new import *

# The following lists prefixed with `filtered_` provide a filtered split
# that:
#
# a. Mitigate a known issue with GTZAN (duplication)
#
# b. Provide a standard split for testing it against other
#    methods (e.g. the one in jordipons/sklearn-audio-transfer-learning).
#
# Those are used when GTZAN is initialised with the `filtered` keyword.
# The split was taken from (github) jordipons/sklearn-audio-transfer-learning.

gtzan_genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock",
]

URL = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
FOLDER_IN_ARCHIVE = "genres"
_CHECKSUMS = {
    "http://opihi.cs.uvic.ca/sound/genres.tar.gz": "5b3d6dddb579ab49814ab86dba69e7c7"
}


def load_gtzan_item(fileid: str, path: str, ext_audio: str) -> Tuple[Tensor, str]:
    """
    Loads a file from the dataset and returns the raw waveform
    as a Torch Tensor, its sample rate as an integer, and its
    genre as a string.
    """
    # Filenames are of the form label.id, e.g. blues.00078
    label, _ = fileid.split(".")
    # Read wav
    file_audio = os.path.join(path, label, fileid + ext_audio)
    waveform, sample_rate = torchaudio.load(file_audio)
    return waveform, sample_rate, label


class GTZAN(Dataset):
    """Create a Dataset for GTZAN.

    Note:
        Please see http://marsyas.info/downloads/datasets.html if you are planning to use
        this dataset to publish results.

    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from.
            (default: ``"http://opihi.cs.uvic.ca/sound/genres.tar.gz"``)
        folder_in_archive (str, optional): The top-level directory of the dataset.
        download (bool, optional):
            Whether to download the dataset if it is not found at root path. (default: ``False``).
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """

    _ext_audio = ".wav"

    def __init__(
        self,
        root: Union[str, Path],
        url: str = URL,
        folder_in_archive: str = FOLDER_IN_ARCHIVE,
        download: bool = False,
        subset: Optional[str] = None,
    ) -> None:

        # super(GTZAN, self).__init__()

        # Get string representation of 'root' in case Path object is passed
        root = os.fspath(root)

        self.root = root
        self.url = url
        self.folder_in_archive = folder_in_archive
        self.download = download
        self.subset = subset

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from "
            + "{'training', 'validation', 'testing'}."
        )

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive)

        if not os.path.isdir(self._path):
            raise RuntimeError(
                "Dataset not found. Please use `download=True` to download it."
            )

        if self.subset is None:
            # Check every subdirectory under dataset root
            # which has the same name as the genres in
            # GTZAN (e.g. `root_dir'/blues/, `root_dir'/rock, etc.)
            # This lets users remove or move around song files,
            # useful when e.g. they want to use only some of the files
            # in a genre or want to label other files with a different
            # genre.
            self._walker = []

            root = os.path.expanduser(self._path)

            for directory in gtzan_genres:
                fulldir = os.path.join(root, directory)

                if not os.path.exists(fulldir):
                    continue

                songs_in_genre = os.listdir(fulldir)
                songs_in_genre.sort()
                for fname in songs_in_genre:
                    name, ext = os.path.splitext(fname)
                    if ext.lower() == ".wav" and "." in name:
                        # Check whether the file is of the form
                        # `gtzan_genre`.`5 digit number`.wav
                        genre, num = name.split(".")
                        if genre in gtzan_genres and len(num) == 5 and num.isdigit():
                            self._walker.append(name)
        else:
            if self.subset == "training":
                self._walker = filtered_train
            elif self.subset == "validation":
                self._walker = filtered_valid
            elif self.subset == "testing":
                self._walker = filtered_test

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, sample_rate, label)``
        """
        fileid = self._walker[n]
        item = load_gtzan_item(fileid, self._path, self._ext_audio)
        waveform, sample_rate, label = item
        return waveform, sample_rate, label

    def __len__(self) -> int:
        return len(self._walker)

    @property
    def walker(self):
        return self._walker
