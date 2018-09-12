import torch
import numpy as np
import math
import torch.nn.functional as F

c = -0.5 * math.log(2*math.pi)

def log_gmm(x, mean, log_var, eps=1e-8):
    # x: (size, dim)
    # mean: (K, dim)
    # log_var: (K, dim)
    return 
    

# on individual component
def log_normal(x, mean, log_var, eps=1e-8):
    # seems to cause NaN?
    #return c - log_var/2 - (x-mean)**2 / (2*log_var.exp() + eps)
    # more stable
    return c - log_var/2 - 0.5 * ((x-mean)**2) * torch.exp(-log_var)

# on individual component
# on unit var -> logvar = 0
def log_normal_unitvar(x, mean, eps=1e-8):
    # seems to cause NaN?
    #return c - log_var/2 - (x-mean)**2 / (2*log_var.exp() + eps)
    # more stable
    return c - 0.5 * ((x-mean)**2) 

# on individual component
def log_logistic(x, mean, logscale, eps=1e-8):
    scale = torch.exp(logscale)
    y = -(x-mean)/scale
    logp = y - logscale - 2*F.softplus(y)
    return logp

# on individual component
# on unit scale s=1
def log_logistic_unitscale(x, mean, eps=1e-8):
    y = -(x-mean)
    logp = y - 2*F.softplus(y)
    return logp

# discretized logistic
# use cdf to do this
def log_discretized_logistic(x, mean, logscale, binsize, eps=1e-8):
    scale = torch.exp(logscale)
    y = (torch.floor(x/binsize)*binsize - mean)/scale

    # direct computing
    # logp = torch.log(F.sigmoid(y + binsize/scale) - F.sigmoid(y) + eps)
    # another more robust way to compute 
    a = y + binsize / scale
    b = y
    logp = -a - F.softplus(-a) - F.softplus(-b) + torch.log(torch.exp(binsize/scale)-1)
    return logp

def log_discretized_logistic_unitscale(x, mean, binsize, eps=1e-8):
    y = torch.floor(x/binsize)*binsize - mean

    # direct computing
    # logp = torch.log(F.sigmoid(y + binsize) - F.sigmoid(y) + eps)
    # another more robust way to compute 
    a = y + binsize
    b = y
    logp = -a - F.softplus(-a) - F.softplus(-b) + math.log(math.exp(binsize)-1)
    return logp

# on individual component
def log_stdnormal(x):
    return c - x**2 / 2

# much stable than sigmoid + BCE
def BCE_with_logits(input, target):
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    return loss

# on individual component
def log_bernoulli(x, p, eps=1e-8): # p in (0, 1)
    p = torch.clamp(p, eps, 1-eps)
    tmp = (x * torch.log(p) + ( 1-x ) * torch.log1p(-p )).mean()
    if tmp.data[0]!=tmp.data[0]: # nan
        print 'x', x.mean().data[0],
        print '\tp', p.mean().data[0]
    return  x * torch.log(p) + ( 1-x ) * torch.log1p(-p )

def log_linoulli(x, p):
    return torch.log(2*x*p + 2*(1-x)*(1-p))

def arctanh_over_x(x):
    return arctanh(x) / x

def arctanh(x): # x in (-1,1)
    # defined as 0.5 * torch.log(x+1) - 0.5 * torch.log(1-x)
    # use log1p here for better numerical stability
    return torch.log1p(2*x/(1-x)) / 2

# p(x) = C p^x(1-p)^(1-x) for x in [0,1], continuous extention to bernoulli
# on individual component
# C = 2tanh^(-1)(1-2p) / (1-2p)
def log_contoulli(x, p, eps=1e-8):
    logC = torch.log(2 * arctanh(1-2*p) + eps) - torch.log(1-2*p + eps)
    p = torch.clamp(p, eps, 1-eps)
    return x * torch.log(p) + (1-x) * torch.log(1-p) + logC 

# approximation to lgamma
def fastlgamma(x):
    xp3 = 3.0 + x
    logterm = torch.log(x) + torch.log(1.0 + x) + torch.log(2.0 + x)
    res = -2.081061466 - x + 0.0833333 / xp3 - logterm + (2.5 + x) * torch.log(xp3)
    return res

# approximation to digamma
def fastdigamma(x):
    twopx = 2.0 + x
    logterm = twopx.log()
    
    res = - (1.0 + 2.0 * x) / (x * (1.0 + x)) \
          - (13.0 + 6.0 * x) / (12.0 * twopx * twopx) \
          + logterm

    return res

# on individual component
def log_beta(x, alpha, beta):
    # currently lgamma is not implemented for Variables in PyTorch
    # use approximations instead
    #lbeta = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha+beta)
    lbeta = fastlgamma(alpha) + fastlgamma(beta) - fastlgamma(alpha+beta)
    return (alpha-1) * torch.log(x) + (beta-1) * torch.log(1-x) - lbeta

# on individual component
def log_kumaraswamy(x, alpha, beta):
    return (alpha-1) * torch.log(x) + (beta-1) * torch.log1p(-x**alpha) + torch.log(alpha) + torch.log(beta)

def log_mean_exp(A, dim):
    A_max = A.max(dim, keepdim=True)[0] # torch. max returns (value, arg_idx)
    B = torch.log(torch.mean(torch.exp(A - A_max.expand_as(A)), dim=dim, keepdim=True)) + A_max # B same shape with A
    B = B.squeeze(dim) # squeeze the resulting dim
    return B

def log_sum_exp(A, dim):
    A_max = A.max(dim, keepdim=True)[0]
    B = torch.log(torch.sum(torch.exp(A - A_max.expand_as(A)), dim=dim, keepdim=True)) + A_max
    B = B.squeeze(dim)
    return B

# convert xs to a map of images
def convert(xs): # xs: (N * dim)
    N = int(np.sqrt(xs.shape[0]))
    image_width = int(np.sqrt(xs.shape[1]))

    y = np.zeros((N * image_width, N*image_width))

    for i in range(0, N):
        for j in range(0, N):
            y[i*image_width: (i+1)*image_width,j*image_width: (j+1)*image_width] = xs[i*N + j,:].reshape(image_width, image_width)

    return y

if __name__== '__main__':
    a = torch.randn(3, 2, 4)
    b = log_mean_exp(a, 0)
    c = torch.log(torch.mean(torch.exp(a), 0))
    print a
    print 'log_mean_exp', b, c


class BatchLoader():

    def __init__(self, data, batch_size, stop_after_epoch=True):
        self.data = torch.from_numpy(data).float()    # N x dim
        self.reset()
        self.BATCH_SIZE = batch_size
        self.stop_after_epoch = stop_after_epoch  # do not stop after epochs

    def next(self):
        """
        @return: input and target tensors of Batch_size x Dimension x Time
        """
        xs = []

        for _ in xrange(self.BATCH_SIZE):
            if self.curr >= len(self.pi):
                self.reset()
                if self.stop_after_epoch:
                    raise StopIteration()

            i = self.pi[self.curr]
            xs.append(self.data[i, :])
            self.curr += 1

        return torch.stack(xs)

    def __iter__(self):
        return self

    def reset(self):
        self.pi = range(self.data.size(0))
        np.random.shuffle(self.pi)
        self.curr = 0

