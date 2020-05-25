import torch

import torch.nn as nn

from .dependency import FunctionCorrelation


Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

	return nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros')


class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()

        self.moduleOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleTwo = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleThr = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFou = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleFiv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.moduleSix = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )
    

    def forward(self, tensorInput):
        tensorOne = self.moduleOne(tensorInput)
        tensorTwo = self.moduleTwo(tensorOne)
        tensorThr = self.moduleThr(tensorTwo)
        tensorFou = self.moduleFou(tensorThr)
        tensorFiv = self.moduleFiv(tensorFou)
        tensorSix = self.moduleSix(tensorFiv)

        return [ tensorOne, tensorTwo, tensorThr, tensorFou, tensorFiv, tensorSix ]
    

class Matching(nn.Module):
    def __init__(self, intLevel):
        super(Matching, self).__init__()

        self.dblBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

        if intLevel != 2:
            self.moduleFeat = nn.Sequential()

        elif intLevel == 2:
            self.moduleFeat = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        if intLevel == 6:
            self.moduleUpflow = None

        elif intLevel != 6:
            self.moduleUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)

        if intLevel >= 4:
            self.moduleUpcorr = None

        elif intLevel < 4:
            self.moduleUpcorr = nn.ConvTranspose2d(in_channels=49, out_channels=49, kernel_size=4, stride=2, padding=1, bias=False, groups=49)

        self.moduleMain = nn.Sequential(
            nn.Conv2d(in_channels=49, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
        )
    

    def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
        tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
        tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

        if tensorFlow is not None:
            tensorFlow = self.moduleUpflow(tensorFlow)
        
        if tensorFlow is not None:
            tensorFeaturesSecond = Backward(tensorInput=tensorFeaturesSecond, tensorFlow=tensorFlow * self.dblBackward)
        
        if self.moduleUpcorr is None:
            tensorCorrelation = nn.functional.leaky_relu(input=FunctionCorrelation(tensorFirst=tensorFeaturesFirst, tensorSecond=tensorFeaturesSecond, intStride=1), negative_slope=0.1, inplace=False)

        elif self.moduleUpcorr is not None:
            tensorCorrelation = self.moduleUpcorr(nn.functional.leaky_relu(input=FunctionCorrelation(tensorFirst=tensorFeaturesFirst, tensorSecond=tensorFeaturesSecond, intStride=2), negative_slope=0.1, inplace=False))

        return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(tensorCorrelation)
    

class Subpixel(nn.Module):
    def __init__(self, intLevel):
        super(Subpixel, self).__init__()

        self.dblBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

        if intLevel != 2:
            self.moduleFeat = nn.Sequential()
        elif intLevel == 2:
            self.moduleFeat = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        self.moduleMain = nn.Sequential(
            nn.Conv2d(in_channels=[ 0, 0, 130, 130, 194, 258, 386 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
        )
    

    def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
        tensorFeaturesFirst = self.moduleFeat(tensorFeaturesFirst)
        tensorFeaturesSecond = self.moduleFeat(tensorFeaturesSecond)

        if tensorFlow is not None:
            tensorFeaturesSecond = Backward(tensorInput=tensorFeaturesSecond, tensorFlow=tensorFlow * self.dblBackward)
        
        return (tensorFlow if tensorFlow is not None else 0.0) + self.moduleMain(torch.cat([ tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow ], 1))
    

class Regularization(nn.Module):
    def __init__(self, intLevel):
        super(Regularization, self).__init__()

        self.dblBackward = [ 0.0, 0.0, 10.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

        self.intUnfold = [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]

        if intLevel >= 5:
            self.moduleFeat = nn.Sequential()
        elif intLevel < 5:
            self.moduleFeat = nn.Sequential(
                nn.Conv2d(in_channels=[ 0, 0, 32, 64, 96, 128, 192 ][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        self.moduleMain = nn.Sequential(
            nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 195 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        if intLevel >= 5:
            self.moduleDist = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
            )
        elif intLevel < 5:
            self.moduleDist = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=([ 0, 0, 7, 5, 5, 3, 3 ][intLevel], 1), stride=1, padding=([ 0, 0, 3, 2, 2, 1, 1 ][intLevel], 0)),
                nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=(1, [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]), stride=1, padding=(0, [ 0, 0, 3, 2, 2, 1, 1 ][intLevel]))
            )

        self.moduleScaleX = nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
        self.moduleScaleY = nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)


    def forward(self, tensorFirst, tensorSecond, tensorFeaturesFirst, tensorFeaturesSecond, tensorFlow):
        tensorDifference = (tensorFirst - Backward(tensorInput=tensorSecond, tensorFlow=tensorFlow * self.dblBackward)).pow(2.0).sum(1, True).sqrt()

        tensorDist = self.moduleDist(self.moduleMain(torch.cat([ tensorDifference, tensorFlow - tensorFlow.view(tensorFlow.size(0), 2, -1).mean(2, True).view(tensorFlow.size(0), 2, 1, 1), self.moduleFeat(tensorFeaturesFirst) ], 1)))
        tensorDist = tensorDist.pow(2.0).neg()
        tensorDist = (tensorDist - tensorDist.max(1, True)[0]).exp()

        tensorDivisor = tensorDist.sum(1, True).reciprocal()

        tensorScaleX = self.moduleScaleX(tensorDist * nn.functional.unfold(input=tensorFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)) * tensorDivisor
        tensorScaleY = self.moduleScaleY(tensorDist * nn.functional.unfold(input=tensorFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tensorDist)) * tensorDivisor

        return torch.cat([ tensorScaleX, tensorScaleY ], 1)
    