import pdb
import torch
import torch.nn as nn

from config import benchmark
from layer import Flatten, ResConv2d, ResConvTranspose2d


class Generator(nn.Module):
    def __init__(self, in_channels=5, out_channels=3):
        super(Generator, self).__init__()
        self.act = nn.Tanh()

        self.conv1 = ResConv2d(in_channels, 32)
        self.conv2 = ResConv2d(32, 64)
        self.conv3 = ResConv2d(64, 128)

        self.res1 = ResConv2d(128, 128, stride=1)
        self.res2 = ResConv2d(128, 128, stride=1)
        self.res3 = ResConv2d(128, 128, stride=1)
        self.res4 = ResConv2d(128, 128, stride=1)
        self.res5 = ResConv2d(128, 128, stride=1)

        if benchmark == 'Avenue':
            self.deconv3 = ResConvTranspose2d(128, 64, output_padding=(1, 0))
            self.deconv2 = ResConvTranspose2d(64, 32)
            self.deconv1 = ResConvTranspose2d(32, out_channels)
        elif benchmark == 'ShanghaiTech':
            self.deconv3 = ResConvTranspose2d(128, 64, output_padding=(1, 1))
            self.deconv2 = ResConvTranspose2d(64, 32, output_padding=(1, 1))
            self.deconv1 = ResConvTranspose2d(32, out_channels)
        elif benchmark == 'ShanghaiTech1':
            self.deconv3 = ResConvTranspose2d(128, 64, output_padding=(1, 1))
            self.deconv2 = ResConvTranspose2d(64, 32, output_padding=(1, 1))
            self.deconv1 = ResConvTranspose2d(32, out_channels)
        elif benchmark == 'ShanghaiTechn':
            self.deconv3 = ResConvTranspose2d(128, 64, output_padding=(1, 1))
            self.deconv2 = ResConvTranspose2d(64, 32, output_padding=(1, 1))
            self.deconv1 = ResConvTranspose2d(32, out_channels)
        elif benchmark == 'UMN':
            self.deconv3 = ResConvTranspose2d(128, 64)
            self.deconv2 = ResConvTranspose2d(64, 32)
            self.deconv1 = ResConvTranspose2d(32, out_channels)
        elif benchmark == 'UMN_split':
            self.deconv3 = ResConvTranspose2d(128, 64)
            self.deconv2 = ResConvTranspose2d(64, 32)
            self.deconv1 = ResConvTranspose2d(32, out_channels)
        elif benchmark == 'UCSDped2':
            self.deconv3 = ResConvTranspose2d(128, 64)
            self.deconv2 = ResConvTranspose2d(64, 32)
            self.deconv1 = ResConvTranspose2d(32, out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)

        x = self.deconv3(x)
        x = self.deconv2(x)
        x = self.deconv1(x)

        return self.act(x)


class RDiscriminator(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=1024):
        super(RDiscriminator, self).__init__()
        self.model = nn.Sequential(ResConv2d(in_channels, 32), ResConv2d(32, 32), ResConv2d(32, 16), Flatten(),
                                   nn.Linear(hidden_channels, 512), nn.LeakyReLU(0.2), nn.Linear(512, 64),
                                   nn.LeakyReLU(0.2), nn.Linear(64, 1))

    def forward(self, x):
        return self.model(x)


class SDiscriminator(nn.Module):
    def __init__(self, in_channels=5):
        super(SDiscriminator, self).__init__()
        self.model = nn.Sequential(ResConv2d(in_channels, 64), ResConv2d(64, 128), ResConv2d(128, 256),
                                   ResConv2d(256, 512), ResConv2d(512, 1024), nn.AdaptiveMaxPool2d((1, 1)), Flatten(),
                                   nn.Linear(1024, 512), nn.LeakyReLU(0.2), nn.Linear(512, 64), nn.LeakyReLU(0.2),
                                   nn.Linear(64, 1))

    def forward(self, x):
        return self.model(x)