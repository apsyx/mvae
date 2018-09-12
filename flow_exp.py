import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import math
import sys

import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils import log_sum_exp
from convflow import CNN_FLOW
from iaflow import IAF_FLOW
from planarflow import PlanarFlow
import SVAE

from utils import *
from cosine_scheduler import CosineAnnealingLR

def log_stdnormal(x):
    c = - 0.5 * math.log(2*math.pi)
    return c - x**2 / 2

def U_z(z, uid=0):
    eps=1e-8
    z1 = z[:, 0]
    z2 = z[:, 1]
    w1z = torch.sin(math.pi/2*z1)
    w2z = 3.0* torch.exp(-0.5*((z1-1)/0.6)**2)
    w3z = 3.0* F.sigmoid((z1-1)/0.3)

    if uid==1: # Potential 1 in NF paper
        tmp = torch.cat(((-0.5*((z1 - 2)/0.6)**2).view(-1, 1), (-0.5*((z1 + 2)/0.6)**2).view(-1,1)), 1)
        return 0.5*((z.norm(p=2, dim=1) - 2)/0.4)**2 - log_sum_exp(tmp, dim=1)
    elif uid==2: # Potentital 2 in NF paper
        return 0.5*((z2-w1z)/0.4)**2 
    elif uid==3: # Potential 3 in NF paper
        tmp = torch.cat(((-0.5*((z2-w1z)/0.35)**2).view(-1, 1),(-0.5*((z2-w1z+w2z)/0.35)**2).view(-1, 1)), 1)
        return -log_sum_exp(tmp, dim=1)
    elif uid==4: # Potential 4 in NF paper
        tmp = torch.cat(((-0.5*((z2-w1z)/0.4)**2).view(-1, 1), (-0.5*((z2-w1z+w3z)/0.35)**2).view(-1, 1)), 1)
        return -log_sum_exp(tmp, dim=1)
    else:
        return 1

def loss_svae(log_qz, z, uid, annealing=1.0):
    loss = log_qz * annealing + U_z(z, uid)
    return loss.mean() # 

def lossf(z0, zK, logdetSum, uid, annealing=1.0):
    # compute loss here
    log_p_z0 = log_stdnormal(z0).sum(1)
    logQbP = (log_p_z0 - logdetSum) * annealing + U_z(zK, uid)
    loss = logQbP.mean() # Reverse KL(q||p)

    #logQ = log_p_z0 - sum_logdet_J
    #loss = tf.reduce_mean(- logQ   * tf.exp(-logQbP)) # - E_p[log Q(x)]
    #loss = tf.reduce_mean(- logQbP * tf.exp(-logQbP)) # Forward KL(p||q)
    #loss = tf.reduce_mean(logQbP - logQbP * tf.exp(-logQbP)) # Reverse + Forward KL(p||q) 

    return loss

def evaluate_bivariate_pdf(p_z, range, npoints):
    """Evaluate (possibly unnormalized) pdf over a meshgrid."""
    side = np.linspace(range[0], range[1], npoints)
    z1, z2 = np.meshgrid(side, side)
    z = np.hstack([z1.reshape(-1, 1), z2.reshape(-1, 1)])

    p_z_fun = p_z(torch.from_numpy(z).cuda())
    
    return z1, z2, p_z_fun.cpu().numpy().reshape((npoints, npoints))


if __name__ == '__main__':
    if  len(sys.argv)!=4:
        print 'Usage: python %s flow_type uid cnn_layers' % (sys.argv[0])
        print '       flow_type: cnn for ConvFlow, iaf for IAF, planar for planar NF, svae for SVAE'
        sys.exit(-1)


    flow_type = sys.argv[1]
    assert flow_type in ['cnn', 'iaf', 'planar', 'svae']
    uid = int(sys.argv[2])
    layers = int(sys.argv[3])
    gpu = False
    if torch.cuda.device_count() > 0:
        gpu = True

    # reproducibility
    #seed = 1234
    #np.random.seed(seed)
    #torch.manual_seed(seed)

    data_dim = 2
    if flow_type == 'cnn':
        model = CNN_FLOW(dim=2, cnn_layers=layers, kernel_size=2)
    elif flow_type == 'iaf':
        model = IAF_FLOW(dim=2, layers=layers)
    elif flow_type == 'planar':
        model = PlanarFlow(dim=2, num_layers=layers)
    elif flow_type == 'svae':
        data_dim = 10
        model = SVAE.Model(data_dim=data_dim, kvs=layers, hidden_dim=2, flow_layers=0, latent_dim=2, noise_dim=2)

    if gpu:
        model = model.cuda()

    print model

    '''
    x=Variable(torch.randn(16, 100))
    print x
    output, logdetJ = model(x)
    print output
    '''

    batch_size = 128
    learning_rate = 0.01
    momentum = 0.9

    #optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum)
    optimizer = torch.optim.RMSprop(model.parameters(), learning_rate) # seems to works much better than ADAM and SGD
    
    max_steps = 20000
    annealing_steps = max_steps * 0.1

    # cosine scheduler
    scheduler = CosineAnnealingLR(optimizer, max_steps)

    model.train()
    for step in xrange(int(max_steps)):
        # optimize
        optimizer.zero_grad()

        if flow_type == 'svae':
            z0 = torch.ones(batch_size, data_dim) # SVAE randomness comes from the last stochastic layer
        else:
            z0 = torch.randn(batch_size, 2)

        if gpu:
            z0 = z0.cuda()
        input = Variable(z0)

        annealing = min(1.0, 1.0*step/annealing_steps)

        if flow_type == 'svae': # SVAE based models
            TIW = 50
            mu, logvar, z, z_f, logdet, x_dec = model(input, TIW)
            # Importance weighted 
            mus     =     mu.unsqueeze(0).permute(1,0,2,3).repeat(1, TIW, 1, 1) 
            logvars = logvar.unsqueeze(0).permute(1,0,2,3).repeat(1, TIW, 1, 1) 
            zs      =      z.unsqueeze(0).repeat(TIW, 1, 1, 1) 
            log_qz  = log_mean_exp(log_normal(zs, mus, logvars).sum(-1), dim=0)

            loss = loss_svae(log_qz.view(-1), z.view(-1, 2), uid, annealing)
        else:
            output, logdetSum = model(input)
            loss = lossf(input, output, logdetSum, uid, annealing)

        loss.backward()#retain_graph=True)
        optimizer.step()
        
        if step % 5 == 0:
            print 'Loss at ITER %d: %f annealing:%f' % (step, loss.data[0], annealing)

        # decrease learning rate every 1000 steps
        scheduler.step()
        
    # plot
    fig = plt.figure()
    

    ax = plt.subplot(1, 3, 1, aspect='equal')
    z1, z2, phat_z = evaluate_bivariate_pdf(lambda z: torch.exp(-U_z(z, uid)), range=(-4, 4), npoints=200)
    z1_true = z1
    z2_true = z2
    phat_z_true = phat_z
    plt.pcolormesh(z1, z2, phat_z)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.invert_yaxis()
    ax.set_title('$p(z)$')
    
    ax = plt.subplot(1, 3, 2, aspect='equal')
    z1, z2, q0_z0 = evaluate_bivariate_pdf(lambda z: torch.exp(log_stdnormal(z).sum(1, keepdim=True)), range=(-4, 4), npoints=200)
    plt.pcolormesh(z1, z2, q0_z0)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.invert_yaxis()
    ax.set_title('$q_0(z)$')

    
    ax = plt.subplot(1, 3, 3, aspect='equal')
    N = int(1e7) # take many samples; but will still look a little 'spotty'

    model.eval()
    if flow_type == 'svae':
        z0_ = np.ones(shape=(N, data_dim))
        mu, logvar, z, z_f, logdet, x_dec = model(Variable(torch.FloatTensor(z0_), volatile=True).cuda())
        zK_ = z.squeeze()
    else:
        z0_ = np.random.normal(size=(N, 2))
        zK_, _ = model(Variable(torch.FloatTensor(z0_), volatile=True).cuda())

    zK_ = zK_.data.cpu().numpy()
    print zK_
    ax.hist2d(zK_[:, 0], zK_[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.invert_yaxis()
    ax.set_title('$z_K \sim q_K(z)$')
    
    fig.tight_layout()
    plt.show()
    plt.savefig('flow_plots/gnf_test_%s_u%d_k%d.pdf' % (flow_type, uid, layers), bbox_inches='tight')
    plt.close()

    
    # plot single image for true and learned
    fig = plt.figure()
    plt.pcolormesh(z1_true, z2_true, phat_z_true)
    plt.axis('off')
    plt.gca().invert_yaxis()
    fig.tight_layout()
    plt.show()
    plt.savefig('flow_plots/gnf_test_%s_u%d_k%d_true.eps' % (flow_type, uid, layers), bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    plt.hist2d(zK_[:, 0], zK_[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
    plt.axis('off')
    plt.gca().invert_yaxis()
    fig.tight_layout()
    plt.show()
    plt.savefig('flow_plots/gnf_test_%s_u%d_k%d_learned.eps' % (flow_type, uid, layers), bbox_inches='tight')
    plt.close()
    
    print('Done :)')
        

