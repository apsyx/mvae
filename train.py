from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from tensorboardX import SummaryWriter
import os
from SVAE import svae_loss_function, svae_iw_loss
from BISVAE import bisvae_loss_function, bisvae_iw_loss
from cosine_scheduler import CosineAnnealingLR
import SVAE 
import BISVAE
from datasets import *
from utils import *
import sys
import datetime

import gzip
import cPickle as pkl
import timeit

parser = argparse.ArgumentParser(description='PyTorch SVAE')
parser.add_argument('--model', type=str, default='SVAE',
                    help='Model: SVAE or BISVAE (default SVAE)')
parser.add_argument("--dataset", type=str, default='fixed',
                    help="sampled or fixed binarized MNIST, sample|fixed|cifar10")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=2000, metavar='N',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status (default: 100)')
parser.add_argument('--test-interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait before testing (default: 100)')
parser.add_argument('--kvs', type=int, default=0,
                    help='Number of auxiliary variables (default: 0)')
parser.add_argument('--tau_dim', type=int, default=50,
                    help='dimension of auxiliary variables (default: 50)')
parser.add_argument('--batchnorm', type=bool, default=True,
                    help='enables bachnorm training')  
parser.add_argument('--lr' ,type=float, default=0.001,
                    help='leanring rate (default: 0.001)')
#not used in cosine LR scheduler
#parser.add_argument('--lr_decay', type=float, default=1.0,
#                    help='learning rate decay (default: 1, no decay. IWAE paper uses 10^{-1/7}=0.719685673001152)')
parser.add_argument('--z_iw', type=int, default=1,
                    help='importance weight of z')
parser.add_argument('--tau_iw', type=int, default=1,
                    help='importance weight of tau')
parser.add_argument("--outfolder", type=str,
                    help="output folder", default='results')
parser.add_argument('--runid', type=str,
                    help='ID for run', default='testrun')
parser.add_argument('--flowtype', type=str, default='convflow',
                    help='Normalizing flow typef (default: convflow)')
parser.add_argument('--flow-layers', type=int, default=0,
                    help='Number of conv flow blocks (default: 0)')
parser.add_argument('--kernel', type=int, default=5,
                    help='Kernel width of 1d conv (default: 5)')
parser.add_argument('--resume', type=int, default=0,
                    help='Resume training from epoch N to continue training (default 0, training from scratch)')
parser.add_argument('--anneal_epoch', type=int, default=0,
                    help='epoch to end annealing kl term in the loss') 
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print (args)

# reproducibility
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

assert args.dataset in ['mnist', 'fixed', 'omniglot', 'cifar10','m', 'f', 'o'], 'data set must be mnist|fixed|omniglot|cifar10'
datestr = datetime.datetime.now().strftime("%Y%m%d_%H%M")
writer = SummaryWriter(log_dir=os.path.join('runs', '%s_%s' % (datestr, args.runid)))

def bernoullisample(x):
    return torch.bernoulli(x)

### LOAD DATA AND SET UP SHARED VARIABLES
normalizer = 1.0
if args.dataset == 'mnist' or args.dataset == 'm':
    print ('Using dynamic MNIST dataset')
    train_x, _ , valid_x, _, test_x, _ = load_mnist_realval()
    train_x = np.concatenate([train_x, valid_x], 0)
    preprocess_data = bernoullisample # DYNAMIC SAMPLING MNIST
elif args.dataset == 'fixed'or args.dataset == 'f':
    print ('Using fixed MNIST dataset')
    train_x, valid_x, test_x = load_mnist_binarized()
    train_x = np.concatenate([train_x, valid_x], 0)
    preprocess_data = lambda dataset: dataset # dummpy function
elif args.dataset == 'omniglot'or args.dataset == 'o':
    print ('Using Omniglot dataset')
    train_x, train_t, train_char, test_x, test_t, test_char = load_omniglot_iwae()
    del train_t, train_char, test_t, test_char
    valid_x = None
    preprocess_data = bernoullisample # DYNAMNIC SAMPLE for OMNIGLOT
elif args.dataset == 'cifar10': # exp on cifar10 not fully supported yet
    print ('Using CIFAR10 dataset')
    train_x, train_y, test_x, test_y, normalizer = load_cifar10()
    N_train = train_x.shape[0]
    N_test = test_x.shape[0]
    train_x = train_x.reshape(N_train, -1)
    test_x = test_x.reshape(N_test, -1)
    preprocess_data = lambda dataset: dataset # dummpy function

    # set normalizer to 1 if use discretized p(x|z)
    normalizer = 1.0

# get dimension of input data
data_dim = train_x.shape[1]

train_loader = BatchLoader(train_x, batch_size = args.batch_size)
test_loader = BatchLoader(test_x, batch_size = 1)


### SET UP LOGFILE AND OUTPUT FOLDER
if not os.path.exists(args.outfolder):
    os.makedirs(args.outfolder)

model = eval(args.model).Model(data_dim=data_dim, kvs=args.kvs, noise_dim=args.tau_dim, batchnorm=args.batchnorm, usecuda = args.cuda, flowtype=args.flowtype, flow_layers=args.flow_layers, kernel_size=args.kernel)
print (model)

if args.resume > 0:
    saved_model_file = '%s/%s.model.epoch%d' % (args.outfolder, args.runid, args.resume)
    if os.path.isfile(saved_model_file):
        print("=> loading model from epoch '{}'".format(saved_model_file))
        model_params = torch.load(saved_model_file)
        model.load_state_dict(model_params)
    else:
        print("=> no saved model found at '{}'".format(saved_model_file))
        sys.exit(-1)


if args.cuda:
    model.cuda()

if args.z_iw > 1:
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-4) #larger eps for stability in IW loss 
else:
    optimizer = optim.Adam(model.parameters(), lr=args.lr) # 
#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9) # doesn't work, NaN
#optimizer = optim.RMSprop(model.parameters(), lr=args.lr) # worse than Adam

# use cosine scheduler
scheduler = CosineAnnealingLR(optimizer, args.epochs)

def train(epoch):
    model.train()
    train_loss = 0
    train_samples = 0
    
    annealing = 1 if args.anneal_epoch == 0 else min(1.0, 1.0*epoch/args.anneal_epoch)

    for batch_idx, data in enumerate(train_loader):
        data = Variable(preprocess_data(data))
        if args.cuda:
            data = data.cuda()

        data_dim = data.size(1)

        optimizer.zero_grad()

        if args.model == 'SVAE':
            mu, logvar, z, z_f, logdet, recon_data = model(data, TIW=args.tau_iw)
            if args.z_iw > 1:
                loss = svae_iw_loss(mu, logvar, z, z_f, logdet, recon_data, data, annealing)
            else:
                loss = svae_loss_function(mu, logvar, z, z_f, logdet, recon_data, data, annealing)
        elif args.model == 'BISVAE':
            z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x = model(data, TIW=args.tau_iw)
            if args.z_iw > 1:
                loss = bisvae_iw_loss(data, z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x, annealing)
            else:
                loss = bisvae_loss_function(data, z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x, annealing)
            
        # deal with NaN and Inf loss
        if np.isnan(loss.data[0]) or np.isinf(loss.data[0]):
            continue  # go to next batch if loss invalid

        train_samples += len(data)
        train_loss += loss.data[0]
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.data),
                100. * batch_idx * len(data) / len(train_loader.data),
                loss.data[0] / len(data)))

    # decrease learning rate as in IWAE paper
    scheduler.step()

    if train_samples == 0:
        return None # none returned for error

    train_NLL = train_loss / train_samples
    train_bitsperdim = (train_NLL / data_dim + math.log(normalizer)) / math.log(2.0)
    print('====> Epoch: {} Average loss: {:.4f} Bits/Dim: {:.4f} on {}/{} samples.\n '.format(
          epoch, train_NLL, train_bitsperdim, train_samples, len(train_loader.data)))
    return train_NLL


def test(epoch):
    model.eval()
    test_loss = 0
    TIW = 128
    for data in test_loader:
        data = Variable(preprocess_data(data), volatile=True)
        if args.cuda:
            data = data.cuda()

        data_dim = data.size(1)
        if args.model == 'SVAE':
            mu, logvar, z, z_f, logdet, recon_data = model(data, TIW)
            test_loss += svae_iw_loss(mu, logvar, z, z_f, logdet, recon_data, data).data[0]
        elif args.model == 'BISVAE':
            z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x = model(data, TIW)
            test_loss += bisvae_iw_loss(data, z1_mu, z1_logvar, z1, z1_f, logdet1, z2_mu, z2_logvar, z2, z2_f, logdet2, p_z1_mu, p_z1_logvar, p_dec_x).data[0]

    test_loss /= len(test_loader.data)
    test_bitsperdim = (test_loss / data_dim + math.log(normalizer)) / math.log(2)
    print('====> Test set loss: {:.4f} Bits/Dim: {:.4f}'.format(test_loss, test_bitsperdim))
    return test_loss


best_loss = float('inf')
best_epoch = -1
train_time = 0
for epoch in range(args.resume + 1, args.epochs + 1):
    start = timeit.default_timer()
    train_loss = train(epoch)
    stop = timeit.default_timer()
    train_time += (stop - start)

    if train_loss!=None:
        writer.add_scalar('train/NLL', train_loss, epoch)

    if epoch % args.test_interval == 0 or epoch == args.epochs:
        test_loss = test(epoch)
        writer.add_scalar('test/NLL', test_loss, epoch)

        if test_loss < best_loss:
            best_loss = test_loss
            best_epoch = epoch

        print('====> Best loss:     {:.4f} at epoch {:d}\n'.format(best_loss, best_epoch))

        # save model
        torch.save(model.state_dict(), '%s/%s.model.epoch%d' % (args.outfolder, args.runid, epoch))

print ('Total training time: {:.4f}\n'.format(train_time))
