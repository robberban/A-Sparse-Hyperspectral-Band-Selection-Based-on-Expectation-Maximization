import torch
from torch import nn
import math
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from torch.autograd import Function


class BS_Layer_with_r(nn.Module):#No Attention, three losses
    def __init__(self, band):
        super(BS_Layer_with_r, self).__init__()
        self.weights = nn.Parameter(torch.zeros(1,band,1,1)+0.5)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()

    def forward(self, X):
        self.x_save = Variable(X,requires_grad=False)
        weights =  self.relu_2(1 - self.relu_1(1 - self.weights))
        out = weights * X
        return out, weights

