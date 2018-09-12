import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import sys

from utils import log_sum_exp

class CNN_FLOW_LAYER(nn.Module):
    def __init__(self, dim, kernel_size, dilation, rescale=True, skip=True):
        super(CNN_FLOW_LAYER, self).__init__()

        self.dim = dim
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.rescale = rescale
        self.skip = skip
        self.usecuda = True

        if self.rescale: # last layer of flow needs to account for the scale of target variable
            self.lmbd = nn.Parameter(torch.FloatTensor(self.dim).normal_().cuda())
        
        self.conv1d = nn.Conv1d(1, 1, kernel_size, dilation=dilation)
            

    def forward(self, x):
        # x is of size (bs x width)
        #kernel_width = 2
        #padding_len = (stride-1) * len + (kernel_size-1)

        # pad x periodically first, then do conv, this is a little complicated
        # padded_x = torch.cat((x, x[:, :(self.kernel_size-1)]), 1)

        # pad zero to the right
        padded_x = F.pad(x, (0, (self.kernel_size-1) * self.dilation))

        # tanh activation
        # activation = F.tanh(self.conv1d(
        # leaky relu activation

        conv1d = self.conv1d(
            padded_x.unsqueeze(1) # to make it (bs, 1, width)
        ).squeeze()

        w = self.conv1d.weight.squeeze()        

        # make sure u[i]w[0] >= -1 to ensure invertibility for h(x)=tanh(x) and with skip
        # tanh
        #activation = F.tanh(conv1d)
        #activation_gradient = 1 - activation**2

        neg_slope = 1e-2
        activation = F.leaky_relu(conv1d, negative_slope=neg_slope)
        activation_gradient = ((activation>=0).float() + (activation<0).float()*neg_slope)

        # for 0<=h'(x)<=1, ensure u*w[0]>-1
        scale = (w[0] == 0).float() * self.lmbd \
                +(w[0] > 0).float() * (-1./w[0] + F.softplus(self.lmbd)) \
                +(w[0] < 0).float() * (-1./w[0] - F.softplus(self.lmbd))


        '''
        activation = F.relu(self.conv1d(
            padded_x.unsqueeze(1) # to make it (bs, 1, width)
        ).squeeze())
        activation_gradient = (activation>=0).float()
        '''
        
        if self.rescale:
            output = activation.mm(torch.diag(scale))
            activation_gradient = activation_gradient.mm(torch.diag(scale))
        else:
            output = activation

        if self.skip:
            output = output + x

        # tanh'(x) = 1 - tanh^2(x)
        # leaky_relu'(x) = 1 if x >0 else 0.01
        # for leaky: leaky_relu_gradient = (output>0).float() + (output<0).float()*0.01
        # tanh
        # activation_gradient = (1 - activation**2).mm(torch.diag(self.lmbd))
        # leaky_relu


        if self.skip:
            logdet = torch.log(torch.abs(activation_gradient*w[0] + 1)).sum(1)
            #logdet = torch.log(torch.abs((activation_gradient*w[0]+1).prod(1) - (activation_gradient*w[1]).prod(1)))
        else:
            logdet = torch.log(torch.abs(activation_gradient*w[0])).sum(1)
            # logdet = torch.log(torch.abs(self.conv1d.weight.squeeze()[0]**self.dim - self.conv1d.weight.squeeze()[1]**self.dim)) + torch.log(torch.abs(nonlinear_gradient)).sum(1)

        return output, logdet

class DILATION_BLOCK(nn.Module):
    def __init__(self, dim, kernel_size):
        super(DILATION_BLOCK, self).__init__()

        self.block = nn.ModuleList()
        i = 0
        while 2**i <= dim:
            conv1d = CNN_FLOW_LAYER(dim, kernel_size, dilation=2**i)
            self.block.append(conv1d)
            i+= 1

    def forward(self, x):
        logdetSum = 0
        output = x
        for i in range(len(self.block)):
            output, logdet = self.block[i](output)
            logdetSum += logdet

        return output, logdetSum
        

class CNN_FLOW(nn.Module):
    def __init__(self, dim, cnn_layers, kernel_size, use_revert=True):
        super(CNN_FLOW, self).__init__()

        # prepare reversion matrix
        # just a matrix whose anti-diagonal are all 1s
        self.usecuda = True
        self.use_revert = use_revert
        self.R = Variable(torch.from_numpy(np.flip(np.eye(dim), axis=1).copy()).float(), requires_grad=False)
        if self.usecuda:
            self.R = self.R.cuda()
        
        self.layers = nn.ModuleList()
        for i in range(cnn_layers):
            #conv1d = CNN_FLOW_LAYER(kernel_size, kernel_size**i)  ## dilation setting
            #            block = CNN_FLOW_LAYER(dim, kernel_size, dilation=1)
            block = DILATION_BLOCK(dim, kernel_size)
            self.layers.append(block)
        
    def forward(self, x):
        logdetSum = 0
        output = x
        for i in range(len(self.layers)):
            output, logdet = self.layers[i](output)
            # revert the dimension of the output after each block
            if self.use_revert:
                output = output.mm(self.R)
            logdetSum += logdet

        return output, logdetSum


