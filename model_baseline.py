# This code is written by Jingyuan Yang @ XD

from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class model_baseline(nn.Module):
    """ResNet50 for Visual Sentiment Analysis on FI_8"""

    def __init__(self, base_model):
        super(model_baseline, self).__init__()
        self.fcn = nn.Sequential(*list(base_model.children())[:-2]) ##-2
        self.GAvgPool = nn.AvgPool2d(kernel_size=14)

        # classifier
        self.classifier8 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=8)
        )

    def forward(self, x):
        x = self.fcn(x)
        x = self.GAvgPool(x)
        x = x.view(x.size(0), x.size(1))

        #-------classifier8--------#
        emotion = self.classifier8(x)

        #-------8to2-------#
        emotion = F.softmax(emotion, dim=1)
        # emotion = torch.nn.LogSoftmax(dim=1)(emotion)

        return emotion