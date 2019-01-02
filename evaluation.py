
from torch.utils.data import DataLoader
import copy
from progress.bar import Bar
import config
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from utils import torch_op
import util
from RPModule.rpmodule import RelativePoseEstimation,getMatchingPrimitive,RelativePoseEstimation_helper
from RPModule.rputil import opts
import argparse
from model.mymodel import SCNet
import time
from baselines import super4pcs, open3d_global_registration, open3d_fast_global_registration,open3d_color_registration
from open3d import *
import logging


def getLoader(args):
    testOption='test'
    if 'suncg' in args.dataList:
        from datasets.SUNCG import SUNCG as Dataset
        dataset_name='suncg'
        val_dataset = Dataset(testOption, nViews=config.nViews,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,list_=f"./data/dataList/{args.dataList}.npy",singleView=0,entrySplit=args.entrySplit)
    elif 'matterport' in args.dataList:
        from datasets.Matterport3D import Matterport3D as Dataset
        dataset_name='matterport'
        val_dataset = Dataset(testOption, nViews=config.nViews,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,list_=f"./data/dataList/{args.dataList}.npy",singleView=0,entrySplit=args.entrySplit)
    elif 'scannet' in args.dataList:
        from datasets.ScanNet import ScanNet as Dataset
        dataset_name='scannet'
        val_dataset = Dataset(testOption, nViews=config.nViews,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,list_=f"./data/dataList/{args.dataList}.npy",singleView=0,fullsize_rgbdn=True,entrySplit=args.entrySplit,representation=args.representation)
    if args.debug:
        loader = DataLoader(val_dataset, batch_size=1, shuffle=False,drop_last=True,collate_fn=util.collate_fn_cat, worker_init_fn=util.worker_init_fn)
    else:
        loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=1,drop_last=True,collate_fn=util.collate_fn_cat, worker_init_fn=util.worker_init_fn)
    return dataset_name,loader

def _parse_args():
    
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--dataList', type = str, default = 'matterport3dv1', help = 'options: suncgv3,scannetv1,matterport3dv1')
    parser.add_argument('--sigmaDist',type=float, default=0.04, help = 'parameter for our pairwise matching algorithm')
    parser.add_argument('--sigmaAngle1',type=float, default=0.2615,help = 'parameter for our pairwise matching algorithm')
    parser.add_argument('--sigmaAngle2',type=float, default=0.2615, help = 'parameter for our pairwise matching algorithm')
    parser.add_argument('--sigmaFeat',type=float, default=0.01, help = 'parameter for our pairwise matching algorithm')
    parser.add_argument('--maxIter',type=int,default=1000, help = 'number of pairs to be tested')
    parser.add_argument('--outputType',type=str,default='rgbdnsf', help = 'types of output')
    parser.add_argument('--debug',action='store_true', help = 'for debug')
    parser.add_argument('--exp',type=str,default='', help = 'will create a folder with such name under experiments/')
    parser.add_argument('--snumclass',type=int,default=15, help = 'number of semantic class')
    parser.add_argument('--featureDim',type=int,default=32, help = 'feature dimension')
    parser.add_argument('--maskMethod',type=str,default='second',help='observe the second view')
    parser.add_argument('--d',type=str,default='', help = '')
    parser.add_argument('--entrySplit',type=int,default=None, help = 'use for parallel eval')
    parser.add_argument('--representation',type=str,default='skybox')
    parser.add_argument('--method',type=str,choices=['ours','ours_nc','ours_nr','super4pcs','fgs','gs','cgs'],default='ours',help='ours,super4pcs,fgs(fast global registration)')
    parser.add_argument('--useTanh', type = int, default = 1, help = 'whether to use tanh layer on feature maps')
    parser.add_argument('--saveCompletion', type = int, default = 1, help = 'save the completion result')
    parser.add_argument('--batchnorm', type = int, default = 1, help = 'whether to use batch norm in completion network')
    parser.add_argument('--skipLayer', type = int, default = 1, help = 'whether to use skil connection in completion network')
    parser.add_argument('--num_repeat', type = int, default = 1, help = 'repeat times')
    parser.add_argument('--rm',action='store_true',help='will remove previous evaluation named args.exp')
    parser.add_argument('--para', type = str, default=None,help = 'file specify parameters for pairwise matching module')
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")

    args = parser.parse_args()
    if args.d: os.environ["CUDA_VISIBLE_DEVICES"] = args.d
    args.alterStep = 1 if args.method == 'ours_nr' else 3
    args.completion = 0 if args.method == 'ours_nc' else 1
    args.snumclass = 15 if 'suncg' in args.dataList else 21
    if args.logLevel:
        logging.basicConfig(level=getattr(logging, args.logLevel))

    print("\n parameters... *******************************\n")
    print(f"evaluate on {args.dataList}")
    print(f"using method: {args.method}")
    print(f"mask method: {args.maskMethod}")
    if 'ours' in args.method:
        print(f"output type: {args.outputType}")
        print(f"semantic classes: {args.snumclass}")
        print(f"feature dimension: {args.featureDim}")
        print(f"skipLayer: {args.skipLayer}")
    print("\n parameters... *******************************\n")
    time.sleep(5)


    args.rpm_para = opts()
    
    args.perStepPara = False
    if args.para is not None:
        para_val = np.loadtxt(args.para).reshape(-1,4)
        args.rpm_para.sigmaAngle1 = para_val[:,0]
        args.rpm_para.sigmaAngle2 = para_val[:,1]
        args.rpm_para.sigmaDist = para_val[:,2]
        args.rpm_para.sigmaFeat = para_val[:,3]
        args.perStepPara = True
    else:
        if args.sigmaAngle1: args.rpm_para.sigmaAngle1 = args.sigmaAngle1
        if args.sigmaAngle2: args.rpm_para.sigmaAngle2 = args.sigmaAngle2
        if args.sigmaDist: args.rpm_para.sigmaDist = args.sigmaDist
        if args.sigmaFeat: args.rpm_para.sigmaFeat = args.sigmaFeat

    return args

if __name__ == '__main__':
    
    args = _parse_args()
    log = logging.getLogger(__name__)

    exp_dir = f"tmp/rpe/{args.exp}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    dataset_name,loader = getLoader(args)
    bar = Bar('Progress', max=len(loader))

    speedBenchmark=[]
    Overlaps = ['0-0.1','0.1-0.5','0.5-1.0']
    adstatsOverlaps = {it:[] for it in Overlaps}
    transstatsOverlaps = {it:[] for it in Overlaps}
    error_stats=[]
    if not args.rm:
        if os.path.exists(f"{exp_dir}/{args.exp}.result.npy"):
            error_stats+=np.load(f"{exp_dir}/{args.exp}.result.npy").tolist()
    n_run = len(error_stats)//100
    args.num_repeat -= n_run
    
    if 'ours' in args.method:
        # setup division point of outputs
        args.idx_f_start = 3+3+1+args.snumclass
        args.idx_f_end   = args.idx_f_start + args.featureDim

        # initialize network and load checkpoint
        net=SCNet(args).cuda()
        try:
            if 'suncg' in args.dataList:
                checkpoint = torch.load('./data/pretrained_model/suncg.comp.pth.tar')
            elif 'matterport' in args.dataList:
                checkpoint = torch.load('./data/pretrained_model/matterport.comp.pth.tar')
            elif 'scannet' in args.dataList:
                checkpoint = torch.load('./data/pretrained_model/scannet.comp.pth.tar')
        except:
            raise Exception("please provide the pretrained model.")

        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        net.cuda()

    for _ in range(args.num_repeat):

        for i, data in enumerate(loader):
            st = time.time()
            np.random.seed()

            # initialize data
            rgb,depth,R,Q,norm,imgPath,segm=data['rgb'],data['depth'],data['R'],data['Q'],data['norm'],data['imgsPath'],data['segm']
            # use origin size scan for baselines on scannet dataset 
            if 'scannet' in args.dataList and 'ours' not in args.method:
                rgb,depth = data['rgb_full'], data['depth_full']
            R     = torch_op.npy(R)
            rgb   = torch_op.npy(rgb*255).clip(0,255).astype('uint8')
            norm  = torch_op.npy(norm)
            depth = torch_op.npy(depth)
            segm  = torch_op.npy(segm)
            
            R_src = R[0,0,:,:]
            R_tgt = R[0,1,:,:]
            R_gt_44 = np.matmul(R_tgt,np.linalg.inv(R_src))
            R_gt = R_gt_44[:3,:3]

            # generate source/target scans, point cloud
            depth_src,depth_tgt,normal_src,normal_tgt,color_src,color_tgt,pc_src,pc_tgt = util.parse_data(depth,rgb,norm,args.dataList,args.method)

            if len(pc_src) == 0 or len(pc_tgt)==0:
                print(f"this point cloud file contain no point")
                continue

            # compute overlap and other stats
            overlap_val,cam_dist_this,pc_dist_this,pc_nn = util.point_cloud_overlap(pc_src, pc_tgt, R_gt_44)
            overlap = '0-0.1' if overlap_val <= 0.1 else '0.1-0.5' if overlap_val <= 0.5 else '0.5-1.0'

            # do not test non-overlap with traditional method since make no sense.
            if args.method in ['fgs','gs','super4pcs','cgs'] and overlap_val < 0.1:
                continue

            # select which method to evaluate
            if args.method == 'super4pcs':
                R_hat = super4pcs(pc_src, pc_tgt)
            elif args.method == 'fgs':
                R_hat = open3d_fast_global_registration(pc_src,pc_tgt)
            elif args.method == 'gs':
                R_hat = open3d_global_registration(pc_src,pc_tgt)
            elif args.method == 'cgs':
                R_hat = open3d_color_registration(pc_src,pc_tgt, color_src,color_tgt)
            elif 'ours' in args.method:
                with torch.set_grad_enabled(False):

                    data_s = {'rgb':   rgb[0,0,:,:,:].transpose(1,2,0),
                            'depth': depth[0,0,:,:],
                            'normal':norm[0,0,:,:,:].transpose(1,2,0),
                            'R':     R[0,0,:,:]}
                    data_t = {'rgb':   rgb[0,1,:,:,:].transpose(1,2,0),
                            'depth': depth[0,1,:,:],
                            'normal':norm[0,1,:,:,:].transpose(1,2,0),
                            'R':     R[0,1,:,:]}

                    R_hat = np.eye(4)

                    # get the complete scans
                    complete_s=torch.cat((torch_op.v(data['rgb'][:,0,:,:,:]),torch_op.v(data['norm'][:,0,:,:,:]),torch_op.v(data['depth'][:,0:1,:,:])),1)
                    complete_t=torch.cat((torch_op.v(data['rgb'][:,1,:,:,:]),torch_op.v(data['norm'][:,1,:,:,:]),torch_op.v(data['depth'][:,1:2,:,:])),1)

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
                        view_t2s=torch_op.v(util.warping(torch_op.npy(view_t),np.linalg.inv(R_hat),args.dataList))
                        view_s2t=torch_op.v(util.warping(torch_op.npy(view_s),R_hat,args.dataList))
                        # append the warped scans
                        view0 = torch.cat((view_s,view_t2s),1)
                        view1 = torch.cat((view_t,view_s2t),1)

                        # generate complete scans
                        f=net(torch.cat((view0,view1)))
                        f0=f[0:1,:,:,:]
                        f1=f[1:2,:,:,:]
                        
                        data_sc,data_tc={},{}
                        # replace the observed region with observed depth/normal
                        data_sc['normal'] = (1-mask_s)*torch_op.npy(f0[0,3:6,:,:]).transpose(1,2,0)+mask_s*data_s['normal']
                        data_tc['normal'] = (1-mask_t)*torch_op.npy(f1[0,3:6,:,:]).transpose(1,2,0)+mask_t*data_t['normal']
                        data_sc['normal']/= (np.linalg.norm(data_s['normal'],axis=2,keepdims=True)+1e-6)
                        data_tc['normal']/= (np.linalg.norm(data_t['normal'],axis=2,keepdims=True)+1e-6)
                        data_sc['depth']  = (1-mask_s[:,:,0])*torch_op.npy(f0[0,6,:,:])+mask_s[:,:,0]*data_s['depth']
                        data_tc['depth']  = (1-mask_t[:,:,0])*torch_op.npy(f1[0,6,:,:])+mask_t[:,:,0]*data_t['depth']
                        data_sc['obs_mask']   = mask_s.copy()
                        data_tc['obs_mask']   = mask_t.copy()
                        data_sc['rgb']    = (mask_s*data_s['rgb']).astype('uint8')
                        data_tc['rgb']    = (mask_t*data_t['rgb']).astype('uint8')
                        
                        # for scannet, we use the original size rgb image(480x640) to extract sift keypoint
                        if 'scannet' in args.dataList:
                            data_sc['rgb_full'] = (torch_op.npy(data['rgb_full'][0,0,:,:,:])*255).astype('uint8')
                            data_tc['rgb_full'] = (torch_op.npy(data['rgb_full'][0,1,:,:,:])*255).astype('uint8')
                            data_sc['depth_full'] = torch_op.npy(data['depth_full'][0,0,:,:])
                            data_tc['depth_full'] = torch_op.npy(data['depth_full'][0,1,:,:])
                    
                        # extract feature maps
                        f0_feat=f0[:,args.idx_f_start:args.idx_f_end,:,:]
                        f1_feat=f1[:,args.idx_f_start:args.idx_f_end,:,:]
                        data_sc['feat']=f0_feat.squeeze(0)
                        data_tc['feat']=f1_feat.squeeze(0)

                        # run relative pose module to get next estimate
                        if args.perStepPara:
                            para_this = opts(args.rpm_para.sigmaAngle1[alter_],args.rpm_para.sigmaAngle2[alter_],args.rpm_para.sigmaDist[alter_],args.rpm_para.sigmaFeat[alter_])
                        else:
                            para_this = args.rpm_para

                        pts3d,ptt3d,ptsns,ptsnt,dess,dest,ptsW,pttW = getMatchingPrimitive(data_sc,data_tc,dataset_name,args.representation,args.completion)
                        # early return if too few keypoint detected
                        if pts3d is None or ptt3d is None or pts3d.shape[0]<2 or pts3d.shape[0]<2:
                            logging.info(f"no pts detected or less than 2 keypoint detected, return identity: {np.eye(3)}")
                            R_hat = np.eye(4)
                        else:
                            R_hat = RelativePoseEstimation_helper({'pc':pts3d.T,'normal':ptsns,'feat':dess,'weight':ptsW},{'pc':ptt3d.T,'normal':ptsnt,'feat':dest,'weight':pttW},para_this)

            # average speed
            time_this = time.time()-st
            speedBenchmark.append(time_this)
            
            # compute rotation error and translation error
            t_hat = R_hat[:3,3]
            R_hat = R_hat[:3,:3]
            
            ad_this = util.angular_distance_np(R_hat, R_gt[np.newaxis,:,:])[0]
            ad_blind_this = util.angular_distance_np(R_gt[np.newaxis,:,:],np.eye(3)[np.newaxis,:,:])[0]
            translation_this = np.linalg.norm(np.matmul((R_hat - R_gt_44[:3,:3]),pc_src.mean(0).reshape(3)) + t_hat - R_gt_44[:3,3])
            translation_blind_this = np.linalg.norm(t_hat - R_gt_44[:3,3])

            # save result for this pair
            R_pred_44=np.eye(4)
            R_pred_44[:3,:3]=R_hat
            R_pred_44[:3,3]=t_hat
            error_stats.append({'img_src':imgPath[0][0],'img_tgt':imgPath[1][0], 'err_ad':ad_this,
                'err_t':translation_this,'err_blind':ad_blind_this,'err_t_blind':translation_blind_this,'overlap':overlap_val,'pc_dist':pc_dist_this,
                'cam_dist':cam_dist_this,'pc_nearest':pc_nn,'R_gt':R_gt_44,'R_pred_44':R_pred_44})
            
            # update statics
            adstatsOverlaps[overlap].append(ad_this)
            transstatsOverlaps[overlap].append(translation_this)

            # print log
            log.info(f"average processing time per pair: {np.sum(speedBenchmark)/len(speedBenchmark)}")
            log.info(f"imgPath:{imgPath},R_hat:{R_hat}")
            log.info(f"ad/ad_blind this :{ad_this}/{ad_blind_this}\n")

            # print progress bar
            Bar.suffix = '{dataset:10}: [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:}'.format(i, len(loader), total=bar.elapsed_td, eta=bar.eta_td,dataset=dataset_name)
            bar.next()
            if (i+1) % 100 == 0:
                np.save(f"{exp_dir}/{args.exp}.result.npy",error_stats)
                sss=''
                for overlap in Overlaps:
                    sss += f"rotation, overlap:{overlap},nobs:{len(adstatsOverlaps[overlap])}, mean:{np.mean(adstatsOverlaps[overlap])} "
                print(sss)
                sss=''
                for overlap in Overlaps:
                    sss += f"translation, overlap:{overlap},nobs:{len(transstatsOverlaps[overlap])}, mean:{np.mean(transstatsOverlaps[overlap])} "
                print(sss)

            if i == args.maxIter:
                break

    np.save(f"{exp_dir}/{args.exp}.result.npy",error_stats)
