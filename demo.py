from xxMusic.Models import *
from scipy import signal
import scipy.io.wavfile
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

filename = '_data/GTZAN/genres/country/country.00003.wav'

# Read wavefile
_, data = scipy.io.wavfile.read(filename)
# data = data[24000:72000] np.random.normal(len(data)) * 0.02
data = signal.resample(data, 16000 * 30)
data_1 = data + np.random.randint(0, 2000, size=[len(data)])
data_2 = data + np.ones(shape=[len(data)]) * 2000
scipy.io.wavfile.write('xx.wav', 16000, data_1.astype(np.int16))
scipy.io.wavfile.write('yy.wav', 16000, data_2.astype(np.int16))


# Get spectrogram, harmonic spectrogram,
# percussive spectrogram, and Mel-spectrogram
# D = librosa.stft(data, n_fft=512, hop_length=128)
# rp = np.max(np.abs(D))

# D_harmonic, D_percussive = librosa.decompose.hpss(D)
# plt.plot()
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp))
# plt.jet()
# plt.show()
#
# plt.plot()
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_harmonic), ref=rp))
# plt.jet()
# plt.show()
#
# plt.plot()
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(D_percussive), ref=rp))
# plt.jet()
# plt.show()
#
# D = librosa.feature.melspectrogram(y=data, sr=16000, n_mels=160,fmax=8000, n_fft=512,hop_length=128, power=1)
# rp = np.max(np.abs(D))
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(D), ref=rp))
# plt.jet()
# plt.show()

#
# # Get the leared 2D representations from MS-SincResNet
# data = torch.from_numpy(data).float()
# data.unsqueeze_(dim=0)
# data.unsqueeze_(dim=0)
# data = data.cuda()
# _, feat1, feat2, feat3 = model(data)
# feat1.squeeze_()
# feat2.squeeze_()
# feat3.squeeze_()
# feat1 = feat1.detach().cpu().numpy()
# feat2 = feat2.detach().cpu().numpy()
# feat3 = feat3.detach().cpu().numpy()
#
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(feat1), ref=rp))
# plt.jet()
# plt.show()
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(feat2), ref=rp))
# plt.jet()
# plt.show()
# librosa.display.specshow(librosa.amplitude_to_db(np.abs(feat3), ref=rp))
# plt.jet()
# plt.show()
