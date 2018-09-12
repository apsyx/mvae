import numpy as np
import sys
import gzip
import cPickle as pkl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import convert
from datasets import *

if len(sys.argv)!=3:
    print 'Usage: python %s dataset plot_name' % sys.argv[0]
    sys.exit(1)

dataset, plot_name = sys.argv[1], sys.argv[2]

if dataset == 'fixed':
    print ('Using FIXED binarized MNIST data')
    f = gzip.open('data/mnist.pkl.gz', 'rb')
    train_x, valid_x, test_x = pkl.load(f)
    train_x = np.concatenate([train_x, valid_x], 0)
elif dataset == 'omniglot':
    print ('Using Omniglot dataseet')
    train_x, train_t, train_char, test_x, test_t, test_char = load_omniglot_iwae()
    del train_t, train_char, test_t, test_char
    valid_x = None

idx = range(train_x.shape[0])
np.random.shuffle(idx)

x_sample = train_x[idx[:100],:]
x_grid = convert(x_sample)

# plot 
plt.imshow(x_grid, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig(plot_name, bbox_inches='tight')

