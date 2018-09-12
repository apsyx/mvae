import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

# Masked Linear
class MaskedLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(MaskedLinear, self).__init__(*args, **kwargs)
        dim_in, dim_out = self.weight.size()
        # make sure input and output has same dims
        assert dim_in == dim_out
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        self.mask.triu_(diagonal=1)

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedLinear, self).forward(x)

# IAF with autoregressive NN
class IAF_LAYER(nn.Module):
    def __init__(self, dim):
        super(IAF_LAYER, self).__init__()
        
        # 2 layer autoregressive NN for 
        # m and s
        self.auto_nn = nn.Sequential(
            MaskedLinear(dim, dim),
            nn.ReLU(True),
        )

        self.m_net = MaskedLinear(dim, dim)
        self.s_net = MaskedLinear(dim, dim)

    def forward(self, z):
        annout = self.auto_nn(z)
        m = self.m_net(annout)
        s = self.s_net(annout)

        sigma = F.sigmoid(s)

        # output
        new_z = sigma * z + ( 1 - sigma) * m

        # logdet
        logdet = torch.log(sigma).sum(1)

        return new_z, logdet

class IAF_FLOW(nn.Module):
    def __init__(self, dim, layers, use_revert=True):
        super(IAF_FLOW, self).__init__()

        # prepare reversion matrix
        # just a matrix whose anti-diagonal are all 1s
        self.usecuda = True
        self.use_revert = use_revert
        self.R = Variable(torch.from_numpy(np.flip(np.eye(dim), axis=1).copy()).float(), requires_grad=False)
        if self.usecuda:
            self.R = self.R.cuda()
        
        self.layers = nn.ModuleList()
        for i in range(layers):
            layer = IAF_LAYER(dim)
            self.layers.append(layer)
        
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
