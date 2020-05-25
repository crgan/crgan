import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import benchmark
from layer import Flatten, ResConv2d, ResConvTranspose2d, Debug


class Generator(nn.Module):
    def __init__(self, in_channels=5, out_channels=3):
        super(Generator, self).__init__()

        self.encoder = nn.Sequential(ResConv2d(in_channels, 32), ResConv2d(32, 32), ResConv2d(32, 16))
        if benchmark == 'Avenue':
            self.decoder = nn.Sequential(ResConvTranspose2d(16, 32, output_padding=(1, 0)), ResConvTranspose2d(32, 32),
                                         ResConvTranspose2d(32, out_channels), nn.Tanh())
        elif benchmark == 'ShanghaiTech':
            self.decoder = nn.Sequential(ResConvTranspose2d(16, 32, output_padding=(1, 1)),
                                         ResConvTranspose2d(32, 32, output_padding=(1, 1)),
                                         ResConvTranspose2d(32, out_channels), nn.Tanh())
        elif benchmark == 'ShanghaiTech1':
            self.decoder = nn.Sequential(ResConvTranspose2d(16, 32, output_padding=(1, 1)),
                                         ResConvTranspose2d(32, 32, output_padding=(1, 1)),
                                         ResConvTranspose2d(32, out_channels), nn.Tanh())
        elif benchmark == 'UMN':
            self.decoder = nn.Sequential(ResConvTranspose2d(16, 32), ResConvTranspose2d(32, 32),
                                         ResConvTranspose2d(32, out_channels), nn.Tanh())
        elif benchmark == 'UCSDped2':
            self.decoder = nn.Sequential(ResConvTranspose2d(16, 32), ResConvTranspose2d(32, 32),
                                         ResConvTranspose2d(32, out_channels), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class RDiscriminator(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=1024):
        super(RDiscriminator, self).__init__()
        self.model = nn.Sequential(ResConv2d(in_channels, 32), ResConv2d(32, 32), ResConv2d(32, 16), Flatten(),
                                   nn.Linear(hidden_channels, 1))

    def forward(self, x):
        return self.model(x)


class SDiscriminator(nn.Module):
    def __init__(self, in_channels=5):
        super(SDiscriminator, self).__init__()
        self.model = nn.Sequential(ResConv2d(in_channels, 64), ResConv2d(64, 128), ResConv2d(128, 256),
                                   ResConv2d(256, 512), nn.AdaptiveMaxPool2d((1, 1)), Flatten(), nn.Linear(512, 1))

    def forward(self, x):
        return self.model(x)