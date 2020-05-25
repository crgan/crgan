import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapePrinter(nn.Module):
    def forward(self, input):
        print(input.shape)
        return input


class Debug(nn.Module):
    def forward(self, input):
        pdb.set_trace()
        return input


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))