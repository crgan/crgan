import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x):

        x = self.conv(x)
        return x


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                                nn.ReLU(inplace=True), nn.BatchNorm2d(out_channels),
                                nn.UpsamplingNearest2d(scale_factor=2))

    def forward(self, x):
        x = self.up(x)
        return x


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
                                  nn.ReLU(inplace=True), nn.BatchNorm2d(out_channels),
                                  nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):
        x = self.down(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, gate_ch, in_channels, inter_ch):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(inter_ch))

        self.W_x = nn.Sequential(nn.Conv2d(in_channels, inter_ch, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(inter_ch))

        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, kernel_size=1, stride=1, padding=0, bias=True),
                                 nn.BatchNorm2d(1), nn.Sigmoid())

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class SqueezeExciteLinear(nn.Module):
    """Squeeze-and-excite context gating for linear
    """
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExciteLinear, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SqueezeExciteConv2d(nn.Module):
    """Squeeze-and-excite context gating for convolution
    """
    def __init__(self, in_channels, reduction=4):
        super(SqueezeExciteConv2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class ResConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, activation='relu', normalization='none'):
        super(ResConv2d, self).__init__()
        self.conv = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]

        if normalization == 'batch':
            self.conv.append(nn.BatchNorm2d(in_channels))
        elif normalization == 'layer':
            self.conv.append(nn.InstanceNorm2d(in_channels, affine=True))

        if activation == 'relu':
            self.conv.append(nn.ReLU(True))
        elif activation == 'leakyrelu':
            self.conv.append(nn.LeakyReLU(0.2, True))

        self.conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

        if normalization == 'batch':
            self.conv.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'layer':
            self.conv.append(nn.InstanceNorm2d(out_channels, affine=True))

        self.conv = nn.Sequential(*self.conv)

        self.res = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(0.2, True)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x) + self.res(x)
        return self.act(x)


class ResConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, output_padding=1, activation='relu', normalization='none'):
        super(ResConvTranspose2d, self).__init__()
        self.conv = [
            nn.ConvTranspose2d(in_channels,
                               out_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               output_padding=output_padding)
        ]

        if normalization == 'batch':
            self.conv.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'instance':
            self.conv.append(nn.InstanceNorm2d(out_channels, affine=True))

        if activation == 'relu':
            self.conv.append(nn.ReLU(True))
        elif activation == 'leakyrelu':
            self.conv.append(nn.LeakyReLU(0.2, True))

        self.conv.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

        if normalization == 'batch':
            self.conv.append(nn.BatchNorm2d(out_channels))
        elif normalization == 'instance':
            self.conv.append(nn.InstanceNorm2d(out_channels, affine=True))

        self.conv = nn.Sequential(*self.conv)

        self.res = nn.ConvTranspose2d(in_channels,
                                      out_channels,
                                      kernel_size=3,
                                      stride=stride,
                                      padding=1,
                                      output_padding=output_padding)

        if activation == 'relu':
            self.act = nn.ReLU(True)
        elif activation == 'leakyrelu':
            self.act = nn.LeakyReLU(0.2, True)

        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv(x) + self.res(x)
        return self.act(x)