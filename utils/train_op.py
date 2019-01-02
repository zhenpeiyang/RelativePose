import numpy as np
import scipy.io as sio
import argparse
import shutil
import os
import platform
from torch.autograd import Variable
import torch
import config
import glob
import sys
import time
import torchvision.utils as vutils

def tboard_add_img(handle,img,title,niter):
    img = vutils.make_grid(img, normalize=False, scale_each=True)
    handle.add_image(title, img, niter)

def debug():
    import ipdb
    ipdb.set_trace()

def env():
    import getpass
    if os.path.exists('/scratch'):
        return 'eldar'
    elif getpass.getuser() == 'zhenpei':
        return 'qhgroup-desktopv'
    elif getpass.getuser() == 'yzp12':
        return 'graphicsai01'
    return

def env_display():
    # whether have display enviroment
    return 'DISPLAY' in os.environ

def import_matplotlib():
    if env_display():
        import matplotlib
    else:
        import matplotlib
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt


def adjust_learning_rate(optimizer, epoch, baseLR, dropLR, DECAY_LIMIT):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = baseLR * (0.5 ** (epoch // dropLR))
    if lr < DECAY_LIMIT:
        lr = DECAY_LIMIT
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    message = 'current learning rate: {0}'.format(lr)
    return message


def get_latest_model(path, identifier):
    models = glob.glob('{0}/*{1}*'.format(path, identifier))
    epoch = [int(model.split('_')[-1].split('.')[0]) for model in models]
    ind = np.array(epoch).argsort()
    models = [models[i] for i in ind]
    return models[-1]

def parse_epoch(path):
    epoch = int(path.split('_')[-1][:-4])
    return epoch

def resume(keyNet, EXP_DIR_PARAMS, key_word):
    try:
        net_path = get_latest_model(EXP_DIR_PARAMS, key_word)
        state = torch.load(net_path)
        keyNet.load_state_dict(state['state_dict'])
        epoch = state['epoch']
        return epoch, net_path, True
    except:
        return None, None, False
        pass

global counting
counting = 0
def variable_hook(grad):
    grad_ = grad.data.cpu().numpy()
    print('variable hook')
    print(np.mean(abs(grad_.flatten())))
    return grad*.1

def parameters_count(net, name):
    
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('total parameters for %s: %d' % (name, params))

def initialize_parser():    
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--rm', action='store_true',
                    help='remove the experiment folder if exists')
    parser.add_argument('--exp', help='add identifier for this experiment')
    parser.add_argument('--param_id', help='parameter identifier')
    parser.add_argument('--resume', action='store_true', help='if specified, resume certain training')
    parser.add_argument('--d', help='specify gpu device')
    parser.add_argument('--g', action='store_true', help='if specified, produce more detailed training log')
    parser.add_argument('--da', help='specify the dataset')
    return parser
    
def initialize_experiment_directories(args):
    args.DETAILED = args.g
    args.EXPERIMENT_INDEX = args.exp if args.exp else '404'
    args.EXP_BASE_DIR = './experiments/exp_' + args.EXPERIMENT_INDEX
    if args.param_id:
        if args.repeat is not None:
            args.EXP_BASE_REPEAT_DIR = os.path.join('./experiments/exp_' + args.EXPERIMENT_INDEX,'repeat_{0}'.format(args.repeat))
            args.EXP_DIR = os.path.join(args.EXP_BASE_REPEAT_DIR, args.param_id)
        else:
            args.EXP_DIR = os.path.join(args.EXP_BASE_DIR, args.param_id)
    else:
        if args.repeat is not None:
            args.EXP_BASE_REPEAT_DIR = os.path.join('./experiments/exp_' + args.EXPERIMENT_INDEX,'repeat_{0}'.format(args.repeat))
            args.EXP_DIR = args.EXP_BASE_REPEAT_DIR
        else:
            args.EXP_DIR = args.EXP_BASE_DIR
    args.EXP_DIR_SAMPLES = args.EXP_DIR + '/samples'
    args.EXP_DIR_PARAMS = args.EXP_DIR + '/params'
    args.EXP_DIR_LOG = os.path.join(args.EXP_DIR, 'exp_{}.csv'.format(args.EXPERIMENT_INDEX))
    if args.d:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.d
    else:
        if env()=='graphicsai01':
            os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'
    for i in range(torch.cuda.device_count()):
        print("detected gpu:{}\n".format(torch.cuda.get_device_name(i)))
    validate_and_execute_arguments(args)
    return args

def platform_specific_initialization(args):
    args.env = env()
    if args.env == 'eldar':
        pass
    elif args.env == 'graphicsai01':
        pass
    elif args.env == 'qhgroup-desktopv':
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
    else:
        pass

def save_config(config, path):
    skip_keys = ['__builtins__', '__doc__', '__name__', '__package__', '__file__']
    keys = dir(config)
    with open(path, 'w') as f:
        for i,key in enumerate(keys):
            if key in skip_keys:
                continue
            value = eval('config.{0}'.format(key))
            f.write('{0} {1}\n'.format(key, value))

def validate_and_execute_arguments(args):
    # cannot set remove and resume both true
    assert(not (args.rm and args.resume))
    # if not specified rm and resume, then the folder must not exists
    if not args.rm and not args.resume:
        assert(not os.path.exists(args.EXP_DIR))
    try:
        if args.rm:
            shutil.rmtree(args.EXP_DIR)
    except:
        pass
    try:
        if args.repeat or args.param_id:
            if not os.path.exists(args.EXP_BASE_DIR):
                os.makedirs(args.EXP_BASE_DIR)
    except:
        pass
    try:
        if args.repeat:
            if not os.path.exists(args.EXP_BASE_REPEAT_DIR):
                os.makedirs(args.EXP_BASE_REPEAT_DIR)
    except:
        pass

    try:
        if not os.path.exists(args.EXP_DIR):
            os.makedirs(args.EXP_DIR)
    except:
        pass
    try:
        if not os.path.exists(args.EXP_DIR_SAMPLES):
            os.makedirs(args.EXP_DIR_SAMPLES)
    except:
        pass
    try:
        if not os.path.exists(args.EXP_DIR_PARAMS):
            os.makedirs(args.EXP_DIR_PARAMS)
    except:
        pass


def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


def decay_learning_rate(LEARNING_RATE, decay, DECAY_LIMIT):
    LEARNING_RATE = LEARNING_RATE / decay
    if LEARNING_RATE < DECAY_LIMIT:
        LEARNING_RATE = DECAY_LIMIT
    return LEARNING_RATE
