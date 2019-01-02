import os
import config
from utils import train_op

class opts():
  def __init__(self):
    self.parser = train_op.initialize_parser()
    self.parser.add_argument('--arch', default='resnet18',help='specify the network architecture')
    self.parser.add_argument('--pretrain',type=int,default=1,help='')
    self.parser.add_argument('--debug', action='store_true', help = 'debug mode')
    self.parser.add_argument('--batch_size', type = int, default = 8, help = '')
    self.parser.add_argument('--max_epoch', type = int, default=1000, help = '')
    self.parser.add_argument('--repeat', type = int, help = '')
    self.parser.add_argument('--batchnorm', type = int, default = 1, help = '')
    self.parser.add_argument('--ganloss', type = int, default = 0, help = '')
    self.parser.add_argument('--pnloss', type = int, default = 0, help = '')
    self.parser.add_argument('--single_view', type = int, default = 1, help = '# ouput')
    self.parser.add_argument('--model',help='resume ckpt')
    self.parser.add_argument('--featurelearning', type = int, default = 0, help = '')
    self.parser.add_argument('--maskMethod', type = str, default = 'second', help = '')
    self.parser.add_argument('--ObserveRatio',default=0.5,type=float,help='')
    self.parser.add_argument('--outputType', type = str, default = 'rgbdnsf', help = '')
    self.parser.add_argument('--GeometricWeight', type = int, default = 0, help = '')
    self.parser.add_argument('--objectFreqLoss', type = int, default = 0, help = '')
    self.parser.add_argument('--cbw', type = int, default = 0, help = 'class balanced weighting')
    self.parser.add_argument('--dataList', type = str, default = 'matterport3dv1', help = 'options: suncgv3,scannetv1,matterport3dv1')
    self.parser.add_argument('--representation', type = str, default = 'skybox', help = 'options: skybox')
    self.parser.add_argument('--skipLayer', type = int, default = 1, help = '')
    self.parser.add_argument('--snumclass', type = int, default = 21, help = '')
    self.parser.add_argument('--parallel', type = int, default = 0, help = '')
    self.parser.add_argument('--featureDim', type = int, default = 32, help = '')
    self.parser.add_argument('--dynamicWeighting', type = int, default = 0, help = '')
    self.parser.add_argument('--recurrent', type = int, default = 0, help = '')
    self.parser.add_argument('--resize224', type = int, default = 0, help = '') # 1
    self.parser.add_argument('--featlearnSegm', type = int, default = 0, help = '') # 1
    self.parser.add_argument('--useTanh', type = int, default = 1, help = '') # 1
    self.parser.add_argument('--D', type = float, default = 0.5, help = '') # 1
    
    
  def parse(self):
    self.args = self.parser.parse_args()
    if self.args.debug:
        self.args.num_workers = 1
    else:
        self.args.num_workers = 8
    return self.args

