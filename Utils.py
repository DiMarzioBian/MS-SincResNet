import os
import torchaudio
from torchaudio.datasets.utils import download_url, extract_archive


def data_downloader(dataset):

    if dataset == 'GTZAN':
        root = '_data/GTZAN/'
        url = "http://opihi.cs.uvic.ca/sound/genres.tar.gz"
        _checksums = {"http://opihi.cs.uvic.ca/sound/genres.tar.gz": "5b3d6dddb579ab49814ab86dba69e7c7"}

        archive = os.path.basename(url)
        archive = os.path.join(root, archive)
        if not os.path.isdir(os.path.join(root, 'genres')):
            if not os.path.isfile(archive):
                checksum = _checksums.get(url, None)
                download_url(url, root, hash_value=checksum, hash_type="md5")
            extract_archive(archive)

    elif dataset == 'EBallroom':
        filename = '_data/EBallroom/chacha/chacha.100701.mp3'
        if os.path.exists(filename):
            print('\nExtended Ballroom dataset downloaded.......\n')
        else:
            raise RuntimeError(
                "\nError loading EBallroom. Please run script 'getEBallroom.py.py' to download......\n"
            )

    elif dataset == 'FMA_small':
        filename = '_data/FMA_small/fma_small/000/000002.mp3'
        if os.path.exists(filename):
            print('\nFMA_small dataset downloaded.......\n')
        else:
            raise RuntimeError(
                "\nError loading FMA_small. Please see README.md......\n"
            )

    else:
        raise RuntimeError("\nDataset "+dataset+" not found......\n")


def set_optimizer_lr(optimizer, lr):
    """Sets the learning rate to a specific one"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_optimizer_lr(optimizer):
    """Update the learning rate"""
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.5

