import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarLayer(nn.Module):
    def __init__(self, dim):
        super(PlanarLayer, self).__init__()
        
        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim,1))
        self.u = nn.Parameter(torch.randn(dim,1))
        self.b = nn.Parameter(torch.randn(1)) # scalar

        
    def forward(self, z):
        # z is (batch_size, dim)
        
        # ensure w^Tu >= -1 as in Appedix A.1
        wu = self.w.dot(self.u)
        mwu = -1 + torch.log1p(wu.exp())
        u_hat = self.u + (mwu - wu) * self.w / self.w.norm()
        
        # compute f_z
        h = F.tanh(z.mm(self.w) + self.b)
        f_z = z + h.mm(u_hat.t())
        
        # compute logdet_jacobian
        psi = (1 - h**2).mm(self.w.t())
        psi_u = psi.mm(u_hat)
        
        logdet_jacobian = torch.log(torch.abs(1 + psi_u))
        
        return f_z, logdet_jacobian

class PlanarFlow(nn.Module):
    def __init__(self, dim, num_layers):
        super(PlanarFlow, self).__init__()

        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(PlanarLayer(dim))

            
    def forward(self, x):
        logdetSum = 0
        output = x
        for i in range(len(self.layers)):
            output, logdet = self.layers[i](output)
            logdetSum += logdet

        return output, logdetSum
    
        

'''
def tall_flow(z):
    eps = 1.0e-7

    # z is (batch_size, dim)
    dim = 2
    
    #dim = z.get_shape()[1]
    W = tf.Variable(tf.random_normal([dim, dim]))
    b = tf.Variable(tf.random_normal([dim, 1]))
    u = tf.Variable(tf.random_normal([dim, 1]))

    # construct W_hat: triangular and diagnal no 0
    W_hat = tf.matrix_band_part(W, 0, -1) # upper triangular of W

    # compute f_z
    zWb = tf.matmul(z, W_hat) + tf.transpose(b)
    f_z = z + tf.transpose(u) * tf.tanh(zWb)

    # compute logdet_jacobian
    psi = 1 - tf.tanh(zWb)**2
    logdet = tf.reduce_sum(tf.log(tf.abs(1 + psi * tf.diag_part(W) * tf.transpose(u)) + eps), 1, keep_dims= True)

    return [f_z, logdet]





def triangular_flow(z):
    eps = 1.0e-7

    # z is (batch_size, dim)
    dim = 2
    
    #dim = z.get_shape()[1]
    W = tf.Variable(tf.random_normal([dim, dim]))
    b = tf.Variable(tf.random_normal([dim, 1]))
    u = tf.Variable(tf.random_normal([dim, 1]))

    # construct W_hat: triangular and diagnal no 0
    W_hat = tf.matrix_band_part(W, 0, -1) # upper triangular of W

    # compute f_z
    zWb = tf.matmul(z, W_hat) + tf.transpose(b)
    f_z = z + tf.transpose(u) * tf.tanh(zWb)

    # compute logdet_jacobian
    psi = 1 - tf.tanh(zWb)**2
    logdet = tf.reduce_sum(tf.log(tf.abs(1 + psi * tf.diag_part(W) * tf.transpose(u)) + eps), 1, keep_dims= True)

    return [f_z, logdet]

def mirror_flow(z):
    eps = 1.0e-10
    dim = 2
    bs = 100

    W1 = tf.Variable(tf.random_normal([dim, dim], mean = 0, stddev=1e-2)) # weight for the 1st network
    W2 = tf.Variable(tf.random_normal([dim, dim], mean = 0, stddev=1e-2)) # weight for the 2nd network

    b1 = tf.Variable(tf.random_normal([dim, 1], mean = 0, stddev=1e-2)) # bias for 1st
    b2 = tf.Variable(tf.random_normal([dim, 1], mean = 0, stddev=1e-2)) # bias for 2nd

    zW1 = tf.matmul(z, W1) + tf.transpose(b1)
    zW2 = tf.matmul(z, W2) + tf.transpose(b2)

    f_z = zW1 * zW2

    # calculate the logdetminant of jacobian
    J = tf.reshape(zW2, [bs, dim, 1]) * tf.reshape(tf.transpose(W1), [1, dim, dim]) + \
        tf.reshape(zW1, [bs, dim, 1]) * tf.reshape(tf.transpose(W2), [1, dim, dim])
    #    J = tf.Print(J, [J[0], J[0].get_shape()], message="This is J: ")
    logdet = tf.reshape(tf.log(tf.abs(tf.matrix_determinant(J)) + eps), [bs, 1]) # determinant overflow is a severe problem here!!!

    return f_z, logdet

def linear_ar_flow(zz):
    dim = 2
    bs = 100

    W = tf.Variable(tf.random_normal([dim, dim]))
    b = tf.Variable(tf.random_normal([dim, 1]))

    z = tf.tanh(zz) # non-linear on the input first
    psi = 1 - z**2 # gradient of tanh

    W_hat = tf.batch_matrix_band_part(W, 0, -1) # upper triangular of W, but diagnal maynot necessarily be 0
    #W_hat = W - tf.matrix_band_part(W, -1, 0) # upper triangular of W and diagnal are 0 (auto regressive model)

    # compute f_z
    zWb = tf.matmul(z, W_hat) + tf.transpose(b)
    f_z = zWb

    # compute logdet_jacobian, logdet_jacobian is 0 in this case
    logdet_jacobian = tf.zeros([bs, 1])#tf.reduce_sum(tf.log(tf.abs(psi)), 1, keep_dims=True)

    logdet = tf.reduce_sum(tf.log(tf.abs(psi * tf.diag_part(W_hat))) , 1, keep_dims = True)

    return [f_z, logdet_jacobian]


def triangular_residue_flow(z):
    eps = 0#1.0e-6
    # z is (batch_size, dim)
    dim = 2
    
    #dim = z.get_shape()[1]
    W = tf.Variable(tf.random_normal([dim, dim]))
    b = tf.Variable(tf.random_normal([dim, 1]))
    u = tf.Variable(tf.constant(1.0))

    # construct W_hat: triangular and diagnal no 0
    W_hat = tf.batch_matrix_band_part(W, 0, -1) # upper triangular of W

    # compute f_z
    # zWb = tf.matmul(z, tf.transpose(W_hat)) + tf.transpose(b)
    zWb = tf.matmul(z, W_hat) + tf.transpose(b)

    # BEGIN NONLINEARITY SECTION
    # tanh nonlinearity
    f_z = z + u * tf.tanh(zWb)
    psi = 1 - tf.tanh(zWb)**2

    # exponential linear
    #f_z = z + u * tf.nn.elu(zWb)
    #psi = tf.to_float(tf.greater_equal(zWb, 0)) * 1 + tf.to_float(tf.less(zWb, 0)) * tf.exp(zWb)
    

    # relu
    #f_z = z + u * tf.nn.relu(zWb)
    #psi = tf.to_float(tf.greater_equal(zWb, 0)) * 1
    # END NONLINEARITY SECTION

    
    #logdet = tf.reduce_sum(tf.log(tf.abs(1 + psi * tf.diag_part(W) )), 1, keep_dims = True)
    logdet = tf.reduce_sum(tf.log(eps + tf.abs(1 + u * tf.mul(psi, tf.diag_part(W_hat)) )), 1, keep_dims = True)

    return [f_z, logdet]

def linear_flow(z): #linear layer?
    dim = 2

    W = tf.Variable(tf.random_normal([dim, dim]))
    b = tf.Variable(tf.random_normal([dim, 1]))

    # construct W_hat: triangular and diagnal no 0
    W_hat = tf.batch_matrix_band_part(W, 0, -1) # upper triangular of W

    f_z = tf.matmul(z, W_hat) + tf.transpose(b)

    logdet = tf.reduce_sum(z + tf.log(tf.abs(tf.diag_part(W_hat))) - z, 1, keep_dims = True)

    return [f_z, logdet]

'''


