import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils import *
from convflow import CNN_FLOW
from iaflow import IAF_FLOW

def batchnormlayer(IN_DIM, OUT_DIM):
    return nn.Sequential(
        nn.Linear(IN_DIM, OUT_DIM),
        nn.BatchNorm1d(OUT_DIM)
    )

def normaldenselayer(IN_DIM, OUT_DIM):
    return nn.Linear(IN_DIM, OUT_DIM)

class Model(nn.Module):
    def __init__(self, data_dim = 784, kvs=0, noise_dim=0, hidden_dim=200, batchnorm = True, usecuda = True, flowtype='convflow', flow_layers=0, kernel_size=5, latent_dim=50):
        super(Model, self).__init__()

        # construct taus
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.hidden_dim = hidden_dim
        self.kvs = kvs
        self.batchnorm = batchnorm
        self.usecuda = usecuda
        self.latent_dim = latent_dim # as in IWAE paper

        if self.batchnorm:
            denselayer = batchnormlayer
        else:
            denselayer = normaldenselayer

        for kv in range(self.kvs):
            if kv==0:
                exec('self.tau_%d_fc = denselayer(%d+%d, %d)' % (kv+1, self.data_dim, self.noise_dim, self.latent_dim))
            else:
                exec('self.tau_%d_fc = denselayer(%d+%d, %d)' % (kv+1, self.latent_dim, self.noise_dim, self.latent_dim))

        
        # ===============================
        self.fc1 = denselayer(self.data_dim + self.latent_dim * self.kvs, self.hidden_dim)
        self.fc2 = denselayer(self.hidden_dim, self.hidden_dim)
        self.fc31 = denselayer(self.hidden_dim, self.latent_dim)
        self.fc32 = denselayer(self.hidden_dim, self.latent_dim)
        
        # set up normalization flows
        assert flowtype in ['convflow', 'iaf']
        self.flowtype = flowtype
        self.flow_layers = flow_layers

        if self.flowtype == 'convflow':
            self.flow = CNN_FLOW(dim=self.latent_dim, cnn_layers=self.flow_layers, kernel_size=kernel_size)
        else:
            self.flow = IAF_FLOW(dim=self.latent_dim, layers=self.flow_layers)

        self.fc4 = denselayer(self.latent_dim, self.hidden_dim)
        self.fc5 = denselayer(self.hidden_dim, self.hidden_dim)
        
        # output layer doesn't use batchnorm
        self.fc6 = nn.Linear(self.hidden_dim, self.data_dim)

        self.h = nn.Tanh() # use tanh as default

    def encode(self, x, TIW):
        input = x.repeat(TIW, 1)
        in_list =[input]

        for kv in range(self.kvs): # iterate the number of auxiliary variables
            tau_fc = eval('self.tau_%d_fc' % (kv+1))
            if self.noise_dim >0:
                noise = Variable(torch.randn(input.size(0), self.noise_dim), requires_grad=False)
                if self.usecuda:
                    noise = noise.cuda()

                tau = self.h(tau_fc(torch.cat([input, noise], dim=1)))
            else:
                tau = self.h(tau_fc(input))
            
            input = tau
            in_list += [tau]

        h1 = self.h(self.fc1(torch.cat(in_list, dim=1)))
        h2 = self.h(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)

    def encode_with_zero_noise(self, x, TIW):
        input = x.repeat(TIW, 1)
        in_list =[input]

        for kv in range(self.kvs): # iterate the number of auxiliary variables
            tau_fc = eval('self.tau_%d_fc' % (kv+1))
            if self.noise_dim >0:
                noise = Variable(torch.zeros(input.size(0), self.noise_dim), requires_grad=False)
                if self.usecuda:
                    noise = noise.cuda()

                tau = self.h(tau_fc(torch.cat([input, noise], dim=1)))
            else:
                tau = self.h(tau_fc(input))
            
            input = tau
            in_list += [tau]

        h1 = self.h(self.fc1(torch.cat(in_list, dim=1)))
        h2 = self.h(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.usecuda:
            eps = torch.cuda.FloatTensor(std.size(0), std.size(1)).normal_()
        else:
            eps = torch.FloatTensor(std.size(0), std.size(1)).normal_()
        eps = Variable(eps, requires_grad=False)
        return eps.mul(std.expand_as(eps)).add_(mu.expand_as(eps))

    def decode(self, z): # output logits instead of p
        h4 = self.h(self.fc4(z))
        h5 = self.h(self.fc5(h4))
        return self.fc6(h5)

    def forward(self, x, TIW=1):
        mu, logvar = self.encode(x.view(-1, self.data_dim), TIW=TIW)
        z = self.reparametrize(mu, logvar)

        # flow to flow_layers
        z_f, logdet = self.flow(z)

        if self.flow_layers != 0:
            logdet = logdet.view(TIW, -1)

        return mu.view(TIW, -1, self.latent_dim), logvar.view(TIW, -1, self.latent_dim), z.view(TIW, -1, self.latent_dim), z_f.view(TIW, -1, self.latent_dim), logdet, self.decode(z_f).view(TIW, -1, self.data_dim)


def _svae_compute_loss_terms(mu, logvar, z, z_f, logdet, recon_x, x, annealing):
    # mu, logvar, z, z_f : (TIW, bs, latent_dim)
    # logdet: (TIW, bs)
    # recon_x: (TIW, bs, data_dim)
    # x: (bs, data_dim)
    
    TIW, bs, _ = z.size()
    log_pzf = log_stdnormal(z_f).sum(-1)

    log_px_given_zf = -BCE_with_logits(recon_x, x).sum(-1)
    
    # AVAE core
    mus     =     mu.unsqueeze(0).permute(1,0,2,3).repeat(1, TIW, 1, 1) 
    logvars = logvar.unsqueeze(0).permute(1,0,2,3).repeat(1, TIW, 1, 1) 
    zs      =      z.unsqueeze(0).repeat(TIW, 1, 1, 1) 

    log_qz_given_x = log_mean_exp(log_normal(zs, mus, logvars).sum(-1), dim=0) # back to (TIW, bs, latent_dim); this is correct way to compute log q(z|x), importance weights over taus
    
    log_qzf_given_x = log_qz_given_x - logdet

    LL = log_px_given_zf + annealing * (log_pzf - log_qzf_given_x)

    return LL
            
def svae_loss_function(mu, logvar, z, z_f, logdet, recon_x, x, annealing=1.0):
    LL = _svae_compute_loss_terms(mu, logvar, z, z_f, logdet, recon_x, x, annealing)
    LL = torch.mean(LL, 0)
    return -torch.sum(LL)

def svae_iw_loss(mu, logvar, z, z_f, logdet, recon_x, x, annealing=1.0):
    LL = _svae_compute_loss_terms(mu, logvar, z, z_f, logdet, recon_x, x, annealing)
    LL = log_mean_exp(LL, dim=0)
    return -torch.sum(LL)

def vae_loss_function(mu, logvar, recon_x, x):    
    BCE = nn.BCELoss(size_average=False)(recon_x.squeeze(), x)

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD

