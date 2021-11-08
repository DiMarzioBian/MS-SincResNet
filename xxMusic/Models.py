import math
import torch

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from xxMusic.Metrics import LabelSmoothingLoss
from xxMusic.center_loss import CenterLoss
from xxMusic.triplet_loss import TripletLoss


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    -----
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1, stride=1, padding=0, dilation=1,
                 bias=False, groups=1, min_low_hz=50, min_band_hz=50):
        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % in_channels
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrical)
        if (kernel_size % 2) == 0:
            self.kernel_size = self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        # self.window_ = torch.hamming_window(self.kernel_size)
        # computing only half of the window
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2)))
        self.window_ = 0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        # Due to symmetry, I only need half of the time axes
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        self.n_ = self.n_.to(waveforms.device)
        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high-low)[:, 0]
        
        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET).
        # I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_
        band_pass_center = 2*band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])
        
        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2*band[:, None])
        filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, filters, stride=self.stride, padding=self.padding, dilation=self.dilation, bias=None,
                        groups=1)


class SpatialPyramidPool2D(nn.Module):
    def __init__(self):
        super(SpatialPyramidPool2D, self).__init__()
        self.local_avg_pool = nn.AdaptiveAvgPool2d(output_size=(2, 2))
        self.global_avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, x):
        # local avg pooling, gives 512@2*2 feature
        features_local = self.local_avg_pool(x)
        # global avg pooling, gives 512@1*1 feature
        features_pool = self.global_avg_pool(x)
        # flatten and concatenate
        out1 = features_local.view(features_local.size()[0], -1)
        out2 = features_pool.view(features_pool.size()[0], -1)
        return torch.cat((out1, out2), 1)


class SPP_Resnet(nn.Module):
    def __init__(self, pretrained=True, enable_spp=True):
        super(SPP_Resnet, self).__init__()

        if enable_spp:
            arch = list(models.resnet18(pretrained=pretrained).children())
            self.model = nn.Sequential(
                nn.Sequential(*arch[:-3]),
                arch[-3:-2][0][0],
                nn.Sequential(*list(arch[-3:-2][0][1].children())[:-1]),
                SpatialPyramidPool2D(),
                nn.Linear(2560, 10, bias=True)
            )
        else:
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Linear(512, 10, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x

class xxMusic(nn.Module):
    def __init__(self, opt):
        super(xxMusic, self).__init__()
        self.resnet_pretrained = opt.resnet_pretrained
        self.enable_spp = opt.enable_spp
        self.loss_type = opt.loss_type
        self.layerNorm = nn.LayerNorm([1, 3*opt.sample_rate])
        self.sincNet1 = nn.Sequential(
            SincConv_fast(out_channels=160, kernel_size=251),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024))
        self.sincNet2 = nn.Sequential(
            SincConv_fast(out_channels=160, kernel_size=501),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024))
        self.sincNet3 = nn.Sequential(
            SincConv_fast(out_channels=160, kernel_size=1001),
            nn.BatchNorm1d(160),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1024))
        self.spp_resnet = SPP_Resnet(pretrained=self.resnet_pretrained, enable_spp=self.enable_spp)
        self.calc_loss = None
        self.calc_loss2 = None
        if self.loss_type=='LabelSmooth':
            self.calc_loss = LabelSmoothingLoss(opt.smooth_label, opt.num_label)
        elif self.loss_type=='CrossEntropy':
            self.calc_loss = nn.CrossEntropyLoss()
        elif self.loss_type=='CenterLoss':
            self.calc_loss = LabelSmoothingLoss(opt.smooth_label, opt.num_label)
            self.calc_loss2 = CenterLoss(num_classes=8, feat_dim=10, use_gpu=True)
        elif self.loss_type=='TripletLoss':
            self.calc_loss = LabelSmoothingLoss(opt.smooth_label, opt.num_label)
            self.calc_loss2 =  TripletLoss(margin=1.0, p=2., mining_type='all')

    def forward(self, x):
        """ Feature extraction """
        x = self.layerNorm(x)

        feat1 = self.sincNet1(x)
        feat2 = self.sincNet2(x)
        feat3 = self.sincNet3(x)

        x = torch.cat((feat1.unsqueeze_(dim=1),
                       feat2.unsqueeze_(dim=1),
                       feat3.unsqueeze_(dim=1)), dim=1)
        x = self.spp_resnet(x)
        return x, feat1, feat2, feat3

    def loss(self, wave, y_gt):
        """ Compute loss """
        score_pred, *_ = self.forward(wave)
        loss = self.calc_loss(score_pred, y_gt)
        if self.loss_type=='CenterLoss':
            loss2 = self.calc_loss2(score_pred, y_gt)
            loss = loss + 1e-4*loss2
        elif self.loss_type=='TripletLoss':
            loss2 = self.calc_loss2(score_pred, y_gt)[0]
            loss = loss + loss2
        _, y_pred = torch.max(score_pred, -1)
        num_correct_pred = y_pred.eq(y_gt).sum()
        return loss, num_correct_pred

    def predict(self, wave, y_gt):
        """ Predict data label and compute loss"""
        score_pred, *_ = self.forward(wave)
        loss = self.calc_loss(score_pred, y_gt)
        if self.loss_type == 'CenterLoss':
            loss2 = self.calc_loss2(score_pred, y_gt)
            loss = loss + 1e-4 * loss2
        elif self.loss_type == 'TripletLoss':
            loss2 = self.calc_loss2(score_pred, y_gt)[0]
            loss = loss + loss2
        _, y_pred = torch.max(score_pred, -1)
        num_correct_pred = y_pred.eq(y_gt).sum()
        return loss, num_correct_pred, y_pred

