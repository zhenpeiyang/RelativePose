import torch
from torch.autograd import Variable
import numpy as np

def draw_normal(shape):
    # TODO: why can't I use *shape to unpack?
    return torch.FloatTensor(shape[0],shape[1]).normal_(0, 1).cuda()

def sample_z(mu, logvar, num=None):
    if num is not None:
        eps = Variable(torch.randn(num, config.Z_SIZE)).cuda()
    else:
        eps = Variable(torch.randn(mu.size())).cuda()
    return mu + eps * torch.exp(logvar / 2)

def v(var, cuda=True, volatile=False):
    if type(var) == torch.Tensor or type(var) == torch.DoubleTensor:
        res = Variable(var.float(),volatile=volatile)
    elif type(var) == np.ndarray:
        res = Variable(torch.from_numpy(var).float(),volatile=volatile)
    if cuda:
        res = res.cuda()
    return res

def npy(var):
    return var.data.cpu().numpy()

def variable_hook(grad):
    grad_ = grad.data.cpu().numpy()
    print('variable hook, mean grad: {0}, contain nan: {1}'.format(np.mean(abs(grad_.flatten())), 'True' if \
        np.isnan(grad_).any() else 'False'))
    return grad