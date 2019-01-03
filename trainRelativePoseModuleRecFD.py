import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np 
from RPModule.rpmodule import RelativePoseEstimation_helper,getMatchingPrimitive
from RPModule.rputil import opts
from util import angular_distance_np 
import argparse
import itertools
import util
from torch.utils.data import DataLoader
import os
import torch
from model.mymodel import SCNet
from utils import torch_op
import copy

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--split', type=str,default='val',help='add identifier for this experiment')
parser.add_argument('--dataset', type=str,default='suncg',help='add identifier for this experiment')
parser.add_argument('--exp', type=str,default='sp_param',help='add identifier for this experiment')
parser.add_argument('--rm', action='store_true',help='add identifier for this experiment')
parser.add_argument('--maskMethod',type=str,default='second', help = 'suncg/matterport:second, scannet:kinect')
parser.add_argument('--alterStep', type=int,default=3,help='add identifier for this experiment')

# net specification
parser.add_argument('--batchnorm', type = int, default = 1, help = 'whether to use batch norm in completion network') # 1
parser.add_argument('--useTanh', type = int, default = 1, help = 'whether to use tanh layer on feature maps')
parser.add_argument('--skipLayer', type = int, default = 1, help = 'whether to use skil connection in completion network') # 1
parser.add_argument('--outputType',type=str,default='rgbdnsf', help = 'types of output')
parser.add_argument('--snumclass',type=int,default=15, help = 'number of semantic class')
parser.add_argument('--featureDim',type=int,default=32, help = 'number of semantic class')

# pairwise match specification
parser.add_argument('--representation',type=str,default='skybox')
parser.add_argument('--completion', type = int, default = 1, help = 'whether to use tanh layer on feature maps')
parser.add_argument('--verbose', type = int, default = 0, help = 'whether to use tanh layer on feature maps')
parser.add_argument('--rlevel', type = int, default = 1, help = 'whether to use tanh layer on feature maps')
parser.add_argument('--para_init', type = str, default=None,help = 'whether to use tanh layer on feature maps')

args = parser.parse_args()
args.alterStep = args.rlevel
if args.rm:
    cmd = f"rm {args.exp}.txt"
    os.system(cmd)

# load network
net=SCNet(args).cuda()
if 'suncg' in args.dataset:
    checkpoint = torch.load('./data/pretrained_model/suncg.comp.pth.tar')
elif 'matterport' in args.dataset:
    checkpoint = torch.load('./data/pretrained_model/matterport.comp.pth.tar')
elif 'scannet' in args.dataset:
    checkpoint = torch.load('./data/pretrained_model/scannet.comp.pth.tar')

state_dict = checkpoint['state_dict']
net.load_state_dict(state_dict)
net.cuda()

if args.para_init is not None:
    para_val = np.loadtxt(args.para_init).reshape(-1,4)
else:
    para_val = np.array([0.523/2,0.523/2,0.08/2,0.01]).reshape(1,4)

args.para=opts(para_val[:,0],para_val[:,1],para_val[:,2],para_val[:,3])

if not os.path.exists("./data/relativePoseModule/"):
    os.makedirs("./data/relativePoseModule/")
primitive_file = f"./data/relativePoseModule/final_{args.dataset}_rlevel_{args.rlevel}.npy"
if os.path.exists(primitive_file):
    primitives=np.load(primitive_file)
else:
    if 'suncg' in args.dataset:
        from datasets.SUNCG import SUNCG as Dataset
        dataset_name='suncg'
        val_dataset = Dataset(args.split, nViews=2,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,\
            list_=f"./data/dataList/suncg.npy",singleView=0)
    elif 'matterport' in args.dataset:
        from datasets.Matterport3D import Matterport3D as Dataset
        dataset_name='matterport'
        val_dataset = Dataset(args.split, nViews=2,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,\
            list_=f"./data/dataList/matterport.npy",singleView=0)
    elif 'scannet' in args.dataset:
        from datasets.ScanNet import ScanNet as Dataset
        dataset_name='scannet'
        val_dataset = Dataset(args.split, nViews=2,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,\
            list_=f"./data/dataList/scannet.npy",singleView=0,fullsize_rgbdn=True,\
            representation=args.representation)

    loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=1,drop_last=True,collate_fn=util.collate_fn_cat, worker_init_fn=util.worker_init_fn)
    primitives=[]
    for i, data in enumerate(loader):
        if i>100:break
        R=torch_op.npy(data['R'])
        R_src = R[0,0,:,:]
        R_tgt = R[0,1,:,:]
        R_gt = np.matmul(R_tgt,np.linalg.inv(R_src))
        
        data_s = {'rgb':torch_op.npy(data['rgb'][0,0,:,:,:]).transpose(1,2,0),
                'norm':torch_op.npy(data['norm'][0,0,:,:,:]).transpose(1,2,0),
                'depth':torch_op.npy(data['depth'][0,0,:,:])
        }
        data_t = {'rgb':torch_op.npy(data['rgb'][0,1,:,:,:]).transpose(1,2,0),
                'norm':torch_op.npy(data['norm'][0,1,:,:,:]).transpose(1,2,0),
                'depth':torch_op.npy(data['depth'][0,1,:,:])
        }
        if 'scannet' in args.dataset:
            data_s['rgb_full']=(torch_op.npy(data['rgb_full'][0,0,:,:,:])*255).astype('uint8')
            data_t['rgb_full']=(torch_op.npy(data['rgb_full'][0,1,:,:,:])*255).astype('uint8')
            data_s['depth_full']=torch_op.npy(data['depth_full'][0,0,:,:])
            data_t['depth_full']=torch_op.npy(data['depth_full'][0,1,:,:])

        args.idx_f_start = 0        
        if 'rgb' in args.outputType:
            args.idx_f_start += 3
        if 'n' in args.outputType:
            args.idx_f_start += 3
        if 'd' in args.outputType:
            args.idx_f_start += 1
        if 's' in args.outputType:
            args.idx_f_start += args.snumclass
        if 'f' in args.outputType:
            args.idx_f_end   = args.idx_f_start + args.featureDim

        with torch.set_grad_enabled(False):
            R_hat=np.eye(4)
            
            # get the complete scans
            complete_s=torch.cat((torch_op.v(data_s['rgb']),torch_op.v(data_s['norm']),torch_op.v(data_s['depth']).unsqueeze(2)),2).permute(2,0,1).unsqueeze(0)
            complete_t=torch.cat((torch_op.v(data_t['rgb']),torch_op.v(data_t['norm']),torch_op.v(data_t['depth']).unsqueeze(2)),2).permute(2,0,1).unsqueeze(0)
            
            # apply the observation mask
            view_s,mask_s,_ = util.apply_mask(complete_s.clone(),args.maskMethod)
            view_t,mask_t,_ = util.apply_mask(complete_t.clone(),args.maskMethod)
            mask_s=torch_op.npy(mask_s[0,:,:,:]).transpose(1,2,0)
            mask_t=torch_op.npy(mask_t[0,:,:,:]).transpose(1,2,0)

            # append mask for valid data
            tpmask = (view_s[:,6:7,:,:]!=0).float().cuda()
            view_s=torch.cat((view_s,tpmask),1)
            tpmask = (view_t[:,6:7,:,:]!=0).float().cuda()
            view_t=torch.cat((view_t,tpmask),1)

            
            for alter_ in range(args.alterStep):
                # warp the second scan using current transformation estimation
                view_t2s=torch_op.v(util.warping(torch_op.npy(view_t),np.linalg.inv(R_hat),args.dataset))
                view_s2t=torch_op.v(util.warping(torch_op.npy(view_s),R_hat,args.dataset))
                # append the warped scans
                view0 = torch.cat((view_s,view_t2s),1)
                view1 = torch.cat((view_t,view_s2t),1)

                # generate complete scans
                f=net(torch.cat((view0,view1)))
                f0=f[0:1,:,:,:]
                f1=f[1:2,:,:,:]
                
                data_sc,data_tc={},{}
                # replace the observed region with gt depth/normal
                data_sc['normal'] = (1-mask_s)*torch_op.npy(f0[0,3:6,:,:]).transpose(1,2,0)+mask_s*data_s['norm']
                data_tc['normal'] = (1-mask_t)*torch_op.npy(f1[0,3:6,:,:]).transpose(1,2,0)+mask_t*data_t['norm']
                data_sc['normal']/= (np.linalg.norm(data_sc['normal'],axis=2,keepdims=True)+1e-6)
                data_tc['normal']/= (np.linalg.norm(data_tc['normal'],axis=2,keepdims=True)+1e-6)
                data_sc['depth']  = (1-mask_s[:,:,0])*torch_op.npy(f0[0,6,:,:])+mask_s[:,:,0]*data_s['depth']
                data_tc['depth']  = (1-mask_t[:,:,0])*torch_op.npy(f1[0,6,:,:])+mask_t[:,:,0]*data_t['depth']

                data_sc['obs_mask']   = mask_s.copy()
                data_tc['obs_mask']   = mask_t.copy()
                data_sc['rgb']    = (mask_s*data_s['rgb']*255).astype('uint8')
                data_tc['rgb']    = (mask_t*data_t['rgb']*255).astype('uint8')
                
                # for scannet, we use the original size rgb image(480x640) to extract sift keypoint
                if 'scannet' in args.dataset:
                    data_sc['rgb_full'] = (data_s['rgb_full']*255).astype('uint8')
                    data_tc['rgb_full'] = (data_t['rgb_full']*255).astype('uint8')
                    data_sc['depth_full'] = data_s['depth_full']
                    data_tc['depth_full'] = data_t['depth_full']
                
                # extract feature maps
                f0_feat=f0[:,args.idx_f_start:args.idx_f_end,:,:]
                f1_feat=f1[:,args.idx_f_start:args.idx_f_end,:,:]
                data_sc['feat']=f0_feat.squeeze(0)
                data_tc['feat']=f1_feat.squeeze(0)

                
                # extract matching primitive from image representation
                pts3d,ptt3d,ptsns,ptsnt,dess,dest,ptsW,pttW = getMatchingPrimitive(data_sc,data_tc,args.dataset,args.representation,args.completion)

                # early return if too few keypoint detected
                if pts3d is None or ptt3d is None or pts3d.shape[0]<2 or pts3d.shape[0]<2:
                    continue
                
                if alter_<args.alterStep-1:
                    # initialize default return value as Identity
                    R_hat=np.eye(4)
                    para_this = copy.copy(args.para)
                    para_this.sigmaAngle1 = para_this.sigmaAngle1[alter_]
                    para_this.sigmaAngle2 = para_this.sigmaAngle2[alter_]
                    para_this.sigmaDist = para_this.sigmaDist[alter_]
                    para_this.sigmaFeat = para_this.sigmaFeat[alter_]
                    R_hat = RelativePoseEstimation_helper({'pc':pts3d.T,'normal':ptsns,'feat':dess,'weight':ptsW},{'pc':ptt3d.T,'normal':ptsnt,'feat':dest,'weight':pttW},para_this)

            dataDict={'pc_src':pts3d.T,'normal_src':ptsns,'feat_src':dess,'weight_src':ptsW,\
                'pc_tgt':ptt3d.T,'normal_tgt':ptsnt,'feat_tgt':dest,'weight_tgt':pttW,'R_gt':R_gt}
            primitives.append(dataDict)
            print(f"{i}/{len(loader)}")

    np.save(primitive_file,primitives)


def objective(para):
    loss = 0
    ad=0
    count = 0
    for i in range(len(primitives)):
        data_s = {'pc':primitives[i]['pc_src'],'normal':primitives[i]['normal_src'],'feat':primitives[i]['feat_src'],'weight':primitives[i]['weight_src']}
        data_t = {'pc':primitives[i]['pc_tgt'],'normal':primitives[i]['normal_tgt'],'feat':primitives[i]['feat_tgt'],'weight':primitives[i]['weight_tgt']}
        R_gt = primitives[i]['R_gt']
        R_hat = RelativePoseEstimation_helper(data_s,data_t,para)

        loss += np.power(R_hat[:3,:3] - R_gt[:3,:3],2).sum()
        ad += angular_distance_np(R_hat[:3,:3].reshape(1,3,3),R_gt[:3,:3].reshape(1,3,3))[0]
        count += 1
        print(i)
        print(ad/count)
    loss /= count
    ad /= count
    print(f"{loss} {ad}")
    return loss,ad

sigmaAngle1_init = 0.523/2
sigmaAngle2_init = 0.523/2
sigmaDist_init = 0.08/2
sigmaFeat_init = 0.01

sigmaAngle1_cur = sigmaAngle1_init
sigmaAngle2_cur = sigmaAngle2_init
sigmaDist_cur   = sigmaDist_init
sigmaFeat_cur   = sigmaFeat_init


for i in range(30):
    N = 10
    epsilon = np.zeros([N,4])
    losses = np.zeros([N])
    ads = np.zeros([N])
    for j in range(N):
        if j>=1:
            epsilon[j,:] = (np.random.uniform(np.zeros([4]))-0.5)/5
        para = opts()
        para.sigmaAngle1 = sigmaAngle1_cur*(1+epsilon[j,0])
        para.sigmaAngle2 = sigmaAngle2_cur*(1+epsilon[j,1])
        para.sigmaDist = sigmaDist_cur*(1+epsilon[j,2])
        para.sigmaFeat = sigmaFeat_cur*(1+epsilon[j,3])
        losses[j],ads[j] = objective(para)
        print(f"loss_{j}:{losses[j]}, ad_{j}:{ads[j]}")

    grad = np.linalg.lstsq(epsilon[1:,:],losses[1:] - losses[0])[0]

    scale = max(np.abs(grad/np.array([sigmaAngle1_cur,sigmaAngle2_cur,sigmaDist_cur,sigmaFeat_cur])))
    grad /= scale

    alpha = 1
    lr=1
    foundDescentDirection = False
    loss_cur = losses[0]
    ad_cur   = ads[0]
    loss_best = loss_cur
    ad_best  = ad_cur

    for j in range(10):
        para = opts()
        para.sigmaAngle1 = sigmaAngle1_cur*(1+-lr*grad*alpha)[0]
        para.sigmaAngle2 = sigmaAngle2_cur*(1+-lr*grad*alpha)[1]
        para.sigmaDist = sigmaDist_cur*(1+-lr*grad*alpha)[2]
        para.sigmaFeat = sigmaFeat_cur*(1+-lr*grad*alpha)[3]
        loss_cur,ad_cur = objective(para)
        if loss_cur < losses[0]:
            sigmaAngle1_cur = para.sigmaAngle1
            sigmaAngle2_cur = para.sigmaAngle2
            sigmaDist_cur = para.sigmaDist
            sigmaFeat_cur = para.sigmaFeat
            foundDescentDirection = True
            break
        alpha /= 2
    if not foundDescentDirection:
        print("cannot found descent direction!\n")
    else:
        print("found descent direction!\n")
        loss_best = loss_cur
        ad_best   = ad_cur
        
    print(f"{loss_best} {ad_best} {sigmaAngle1_cur} {sigmaAngle2_cur} {sigmaDist_cur} {sigmaFeat_cur}")
    with open(f"{args.exp}.txt",'a') as f:
        f.write(f"{loss_best} {ad_best} {sigmaAngle1_cur} {sigmaAngle2_cur} {sigmaDist_cur} {sigmaFeat_cur}\n")

print(losses)



