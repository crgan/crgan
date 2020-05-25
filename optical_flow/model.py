import torch

import torch.nn as nn
import torch.nn.functional as F

from .layer import Features, Matching, Subpixel, Regularization


class LiteFlowNet(nn.Module):
	def __init__(self):
		super(LiteFlowNet, self).__init__()
		self.moduleFeatures = Features()
		self.moduleMatching = nn.ModuleList([ Matching(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
		self.moduleSubpixel = nn.ModuleList([ Subpixel(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])
		self.moduleRegularization = nn.ModuleList([ Regularization(intLevel) for intLevel in [ 2, 3, 4, 5, 6 ] ])

        
	def forward(self, tensorFirst, tensorSecond):
		tensorFirst[:, 0, :, :] = tensorFirst[:, 0, :, :] - 0.411618
		tensorFirst[:, 1, :, :] = tensorFirst[:, 1, :, :] - 0.434631
		tensorFirst[:, 2, :, :] = tensorFirst[:, 2, :, :] - 0.454253

		tensorSecond[:, 0, :, :] = tensorSecond[:, 0, :, :] - 0.410782
		tensorSecond[:, 1, :, :] = tensorSecond[:, 1, :, :] - 0.433645
		tensorSecond[:, 2, :, :] = tensorSecond[:, 2, :, :] - 0.452793

		tensorFeaturesFirst = self.moduleFeatures(tensorFirst)
		tensorFeaturesSecond = self.moduleFeatures(tensorSecond)

		tensorFirst = [ tensorFirst ]
		tensorSecond = [ tensorSecond ]

		for intLevel in [ 1, 2, 3, 4, 5 ]:
			tensorFirst.append(nn.functional.interpolate(input=tensorFirst[-1], size=(tensorFeaturesFirst[intLevel].size(2), tensorFeaturesFirst[intLevel].size(3)), mode='bilinear', align_corners=False))
			tensorSecond.append(nn.functional.interpolate(input=tensorSecond[-1], size=(tensorFeaturesSecond[intLevel].size(2), tensorFeaturesSecond[intLevel].size(3)), mode='bilinear', align_corners=False))
		
		tensorFlow = None

		for intLevel in [ -1, -2, -3, -4, -5 ]:
			tensorFlow = self.moduleMatching[intLevel](tensorFirst[intLevel], tensorSecond[intLevel], tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel], tensorFlow)
			tensorFlow = self.moduleSubpixel[intLevel](tensorFirst[intLevel], tensorSecond[intLevel], tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel], tensorFlow)
			tensorFlow = self.moduleRegularization[intLevel](tensorFirst[intLevel], tensorSecond[intLevel], tensorFeaturesFirst[intLevel], tensorFeaturesSecond[intLevel], tensorFlow)
		
		return tensorFlow * 20.0