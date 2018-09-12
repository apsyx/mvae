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
    def __init__(self, data_dim = 784, kvs=0, noise_dim=0, batchnorm = True, usecuda = True, flowtype='convflow', flow_layers=8, kernel_size=5):
        super(Model, self).__init__()

        # construct taus
        self.data_dim = data_dim
        self.noise_dim = noise_dim
        self.kvs = kvs
        self.batchnorm = batchnorm
        self.usecuda = usecuda

        self.hidden_dim1 = 200 # as in IWAE paper
        self.hidden_dim2 = 100 # as in IWAE 

        self.latent_dim1 = 100 # as in IWAE 
        self.latent_dim2 = 50 # as in IWAE 

        if self.batchnorm:
            denselayer = batchnormlayer
        else:
            denselayer = normaldenselayer


        for kv in range(self.kvs):
            if kv==0:
                exec('self.tau_%d_fc = denselayer(%d+%d, %d)' % (kv+1, self.data_dim, self.noise_dim, self.latent_dim1))
            else:
                exec('self.tau_%d_fc = denselayer(%d+%d, %d)' % (kv+1, self.latent_dim1, self.noise_dim, self.latent_dim1))

        # ===============================
        # layers for z1
        self.fc1 = denselayer(self.data_dim + self.latent_dim1 * self.kvs, self.hidden_dim1)
        self.fc2 = denselayer(self.hidden_dim1, self.hidden_dim1)
        
        self.fc31 = denselayer(self.hidden_dim1, self.latent_dim1)
        self.fc32 = denselayer(self.hidden_dim1, self.latent_dim1)

        # set up normalizing flows
        assert flowtype in ['convflow', 'iaf']
        self.flowtype = flowtype
        self.flow_layers = flow_layers

        if self.flowtype == 'convflow':
            self.flow1 = CNN_FLOW(dim=self.latent_dim1, cnn_layers=self.flow_layers, kernel_size=kernel_size)
        else:
            self.flow1 = IAF_FLOW(dim=self.latent_dim1, layers=self.flow_layers)

        # layers for z2
        self.fc4 = denselayer(self.latent_dim1, self.hidden_dim2)
        self.fc5 = denselayer(self.hidden_dim2, self.hidden_dim2)

        self.fc61 = denselayer(self.hidden_dim2, self.latent_dim2)
        self.fc62 = denselayer(self.hidden_dim2, self.latent_dim2)

        if self.flowtype == 'convflow':
            self.flow2 = CNN_FLOW(dim=self.latent_dim2, cnn_layers=self.flow_layers, kernel_size=kernel_size)
        else:
            self.flow2 = IAF_FLOW(dim=self.latent_dim2, layers=self.flow_layers)

        # decode z2 to z1
        self.fc7 = denselayer(self.latent_dim2, self.hidden_dim2)
        self.fc8 = denselayer(self.hidden_dim2, self.hidden_dim2)

        self.fc91 = denselayer(self.hidden_dim2, self.latent_dim1)
        self.fc92 = denselayer(self.hidden_dim2, self.latent_dim1)

        # dec z1 to x
        self.fc10 = denselayer(self.latent_dim1, self.hidden_dim1)
        self.fc11 = denselayer(self.hidden_dim1, self.hidden_dim1)

        # output layer don't use batchnorm
        self.fc12 = nn.Linear(self.hidden_dim1, self.data_dim)
        
        self.h = nn.Tanh() # use tanh as default activation function

    def encode_x(self, x, TIW):
        input = x.repeat(TIW, 1)
        in_list = [input]

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

    def encode_x_with_zero_noise(self, x, TIW):
        input = x.repeat(TIW, 1)
        in_list = [input]

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

    def encode_z1(self, z1):
        h4 = self.h(self.fc4(z1))
        h5 = self.h(self.fc5(h4))
        return self.fc61(h5), self.fc62(h5)


    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.usecuda:
            eps = torch.cuda.FloatTensor(std.size(0), std.size(1)).normal_()
        else:
            eps = torch.FloatTensor(std.size(0), std.size(1)).normal_()
        eps = Variable(eps, requires_grad=False)
        return eps.mul(std.expand_as(eps)).add_(mu.expand_as(eps))

    def decode_z2(self, z2):
        h7 = self.h(self.fc7(z2))
        h8 = self.h(self.fc8(h7))
        return self.fc91(h8), self.fc92(h8)

    def decode_z1(self, z1): # output logits instead of p
        h10 = self.h(self.fc10(z1))
        h11 = self.h(self.fc11(h10))
        return self.fc12(h11)

    def forward(self, x, TIW=1):
        z1_mu, z1_logvar = self.encode_x(x.view(-1, self.data_dim), TIW=TIW)
        z1 = self.reparametrize(z1_mu, z1_logvar) 

        # flow to cnn_layer1 for z1
        z1_f, logdet1 = self.flow1(z1)

        if self.flow_layers > 0:
            logdet1 = logdet1.view(TIW, -1)

        z2_mu, z2_logvar = self.encode_z1(z1_f)
        z2 = self.reparametrize(z2_mu, z2_logvar) # z2 has same size as z1

        # flow to cnn_layer2 for z2
        z2_f, logdet2 = self.flow2(z2)

        if self.flow_layers > 0:
            logdet2 = logdet2.view(TIW, -1)

        p_z1_mu, p_z1_logvar = self.decode_z2(z2_f)
        p_dec_x = self.decode_z1(z1_f)

        z1_mu = z1_mu.view(TIW, -1, z1_mu.size(-1))
        z1_logvar = z1_logvar.view(TIW, -1, z1_logvar.size(-1))
        z1 = z1.view(TIW, -1, z1.size(-1))
        z1_f = z1_f.view(TIW, -1, z1_f.size(-1))

        z2_mu = z2_mu.view(TIW, -1, z2_mu.size(-1))
        z2_logvar = z2_logvar.view(TIW, -1, z2_logvar.size(-1))
        z2 = z2.view(TIW, -1, z2.size(-1))
        z2_f = z2_f.view(TIW, -1, z2_f.size(-1))

        p_z1_mu = p_z1_mu.view(TIW, -1, p_z1_mu.size(-1))
        p_z1_logvar = p_z1_logvar.view(TIW, -1, p_z1_logvar.size(-1))
        p_dec_x = p_dec_x.view(TIW, -1, p_dec_x.size(-1))
        
        return z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x


def _bisvae_compute_loss_terms(x, z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x, annealing):
    TIW, bs, _ = p_z1_mu.size()
    log_pz2f = log_stdnormal(z2_f).sum(-1) # iw x BS
    log_pz1f_given_z2f = log_normal(z1_f, p_z1_mu, p_z1_logvar).sum(-1) # iw x BS
    log_px_given_z1f = -BCE_with_logits(p_dec_x, x).sum(-1) # more stable
    
    # AVAE core
    z1_mus     =     z1_mu.unsqueeze(0).permute(1,0,2,3).repeat(1, TIW, 1, 1)
    z1_logvars = z1_logvar.unsqueeze(0).permute(1,0,2,3).repeat(1, TIW, 1, 1)
    z1s        =        z1.unsqueeze(0).repeat(TIW, 1, 1, 1)
    log_qz1_given_x = log_mean_exp(log_normal(z1s, z1_mus, z1_logvars).sum(-1), dim=0) # back to (TIW, bs, latent_dim); this is correct way to compute log q(z|x), importance weights over taus

    log_qz1f_given_x = log_qz1_given_x - logdet1 # iw x BS
    log_qz2f_given_z1f = log_normal(z2, z2_mu, z2_logvar).sum(-1) - logdet2 # iw x BS
    LL = log_px_given_z1f + annealing * (log_pz2f + log_pz1f_given_z2f - log_qz1f_given_x - log_qz2f_given_z1f)

    return LL

def bisvae_loss_function(x, z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2,  p_z1_mu, p_z1_logvar, p_dec_x, annealing=1.0):
    LL = _bisvae_compute_loss_terms(x, z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x, annealing)
    LL = torch.mean(LL, 0)
    return -torch.sum(LL)

def bisvae_iw_loss(x, z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x, annealing=1.0):
    LL = _bisvae_compute_loss_terms(x, z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x, annealing)
    LL = log_mean_exp(LL, dim=0)
    return -torch.sum(LL)
