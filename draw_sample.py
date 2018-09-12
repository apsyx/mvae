import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import convert
import SVAE
import BISVAE
from datasets import *
from plot_helpers import plotPCA, plotTSNE, read_from_log, plot_learning_curve

#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

if len(sys.argv)!=3:
    print 'Usage: python %s model_path plot_prefix' % sys.argv[0]
    sys.exit(1)

model_path, prefix = sys.argv[1], sys.argv[2]
noise_dim = 50
kvs = 0
if 'k' in model_path:
    kvs = int(model_path.split('k')[-1][0])
        
if '+' in model_path:
    noise_dim = 0

if 'bivae' in model_path:
    BISVAE_FLAG = True
    model = BISVAE.Model(kvs=kvs, noise_dim=noise_dim, flow_layers=0)
else:
    BISVAE_FLAG = False
    model = SVAE.Model(kvs=kvs, noise_dim=noise_dim, flow_layers=0)

model.load_state_dict(torch.load(model_path))
model.cuda()
model.eval()

# random generated sample
z = Variable(torch.randn(100,50).cuda())              
model.usecuda= True

if BISVAE_FLAG:
    # for BISVAE
    mu1, logvar1 = model.decode_z2(z)
    z1 = model.reparametrize(mu1, logvar1).squeeze()
    x_dec = F.sigmoid(model.decode_z1(z1))
else:
    # for SVAE
    x_dec = F.sigmoid(model.decode(z))

print 'plot samples from model'
#x_sample = torch.bernoulli(x_dec)  # sample 
x_sample = (x_dec>=0.5).int() # just take the mean
x_grid = convert(x_sample.cpu().data.numpy())

plt.imshow(x_grid, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig(prefix+'_sample.eps', bbox_inches='tight')
plt.close()

# need to take care of data sets
if 'f_' in model_path: # mnist model
    train_x, train_y, valid_x, valid_y, test_x, test_y = load_mnist_realval()
    #train_x, valid_x, test_x = load_mnist_binarized()
elif 'o_' in model_path: # omniglot model
    train_x, train_y, train_char, test_y, test_t, test_char = load_omniglot_iwae()

# plot reconstruction for random digits
idx = range(train_x.shape[0])
np.random.shuffle(idx)

x_sample = torch.bernoulli(torch.from_numpy(train_x[idx[:100],:]).float()).numpy() # dynamic 
plt.imshow(convert(x_sample), cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig(prefix+'_real.eps', bbox_inches='tight')
print train_y[idx[:100]]

if BISVAE_FLAG:
    mu1, logvar1 = model.encode_x_with_zero_noise(Variable(torch.from_numpy(x_sample).float().cuda()), 1)
    z1 = mu1 # use mean as encodings
    x_dec = F.sigmoid(model.decode_z1(z1))
else:
    mu, logvar = model.encode(Variable(torch.from_numpy(x_sample).float().cuda()), 1)
    z = mu # use mean
    #z = model.reparametrize(mu, logvar) # real sampling
    x_dec = F.sigmoid(model.decode(z))
    
#x_recon = torch.bernoulli(x_dec)
x_recon = (x_dec>=0.5).int() # just take the mean    
x_grid = convert(x_recon.cpu().data.numpy())

plt.imshow(x_grid, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig(prefix+'_recon.eps', bbox_inches='tight')
plt.close()

