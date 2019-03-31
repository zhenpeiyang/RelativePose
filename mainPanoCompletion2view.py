import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.plot import plotHistogram,plotCummulative,plotSeries
from utils import train_op, torch_op
from utils.torch_op import v,npy
from utils.log import AverageMeter
from utils import log
import config
from tensorboardX import SummaryWriter
import cv2
import util
import time
import re
import glob
from opts import opts
from utils.dotdict import *
from quaternion import *
from utils.factory import trainer
from model.mymodel import Resnet18_8s, SCNet
from utils.callbacks import PeriodicCallback, OnceCallback, ScheduledCallback,CallbackLoc
import copy
from sklearn.decomposition import PCA

#**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--
#**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--
# here is the place for customized functions

def visNorm(vis):
    for v in range(len(vis)):
        if (vis[v].max().item() - vis[v].min().item())!=0:
            vis[v] = (vis[v]-vis[v].min())/(vis[v].max()-vis[v].min())
    return vis

def visNormV1(vis,min_,max_):
    for v in range(len(vis)):
        if (max_ - min_)!=0:
            vis[v] = ((vis[v]-min_)/(max_-min_)).clamp(0,None)
    return vis

def class_to_color(classIdx, dataList):
    if 'scannet' in dataList:
        colors = config.scannet_color_palette[classIdx,:]
    elif 'matterport' in dataList:
        colors = config.matterport_color_palette[classIdx,:]
    elif 'suncg' in dataList:
        colors = config.suncg_color_palette[classIdx,:]
    return colors
def apply_mask(x,maskMethod,*arg):
    # input: [n,c,h,w]
    h=x.shape[2]
    w=x.shape[3]
    tp = np.zeros([x.shape[0],1,x.shape[2],x.shape[3]])
    geow=np.zeros([x.shape[0],1,x.shape[2],x.shape[3]])
    if maskMethod == 'second':
        tp[:,:,:h,h:2*h]=1
        ys,xs=np.meshgrid(range(h),range(w),indexing='ij')
        dist=np.stack((np.abs(xs-h),np.abs(xs-(2*h)),np.abs(xs-w-h),np.abs(xs-w-(2*h))),0)
        dist=dist.min(0)/h
        sigmaGeom=0.7
        dist=np.exp(-dist/(2*sigmaGeom**2))
        dist[:,h:2*h]=0
        geow = torch_op.v(np.tile(np.reshape(dist,[1,1,dist.shape[0],dist.shape[1]]),[geow.shape[0],1,1,1]))
    elif maskMethod == 'kinect':
        assert(w==640 and h==160)
        dw = int(89.67//2)
        dh = int(67.25//2)
        tp[:,:,80-dh:80+dh,160+80-dw:160+80+dw]=1
        geow = tp.copy()*20
        geow[tp==0]=1
        geow = torch_op.v(geow)
    tp=torch_op.v(tp)
    x=x*tp
    return x,tp,geow

#**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--
#**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--

def buildDataset(args):
    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    if 'suncg' in args.dataList:
        from datasets.SUNCG import SUNCG as Dataset
    elif 'scannet' in args.dataList:
        from datasets.ScanNet import ScanNet as Dataset
    elif 'matterport' in args.dataList:
        from datasets.Matterport3D import Matterport3D as Dataset
    else:
        raise Exception("unknown dataset!")

    train_dataset = Dataset('train', config.nViews,AuthenticdepthMap=False,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True\
    ,list_=f"./data/dataList/{args.dataList}.npy",singleView=args.single_view,denseCorres=args.featurelearning,reproj=True,\
    representation=args.representation,dynamicWeighting=args.dynamicWeighting,snumclass=args.snumclass)
    val_dataset = Dataset('test', nViews=config.nViews,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,\
        list_=f"./data/dataList/{args.dataList}.npy",singleView=args.single_view,denseCorres=args.featurelearning,reproj=True,\
        representation=args.representation,dynamicWeighting=args.dynamicWeighting,snumclass=args.snumclass)

    if args.num_workers == 1:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True, collate_fn=util.collate_fn_cat, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True, collate_fn=util.collate_fn_cat, worker_init_fn=worker_init_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True,collate_fn=util.collate_fn_cat, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,drop_last=True,collate_fn=util.collate_fn_cat, worker_init_fn=worker_init_fn)

    return train_loader,val_loader

class learnerParam(object):
    def __init__(self,train_step_vis=600,val_step_vis=50,\
                    train_step_log=100,val_step_log=10):
        self.train_step_vis = train_step_vis
        self.val_step_vis = val_step_vis
        self.train_step_log = train_step_log
        self.val_step_log = val_step_log

class learner(object):
    def __init__(self,args,learnerParam):
        self.learnerParam=learnerParam
        self.args=args
        self.epochStart = 0
        self.userConfig()

        # build network
        if self.args.representation == 'skybox':
            self.netG=SCNet(args).cuda()
            Fargs = copy.copy(args)
            Fargs.num_input = 7
            self.netF=Resnet18_8s(Fargs).cuda()
            
            if 'suncg' in self.args.dataList:
                checkpoint = torch.load('./data/pretrained_model/suncg.feat.pth.tar')
            elif 'matterport' in self.args.dataList:
                checkpoint = torch.load('./data/pretrained_model/matterport.feat.pth.tar')
            elif 'scannet' in self.args.dataList:
                checkpoint = torch.load('./data/pretrained_model/scannet.feat.pth.tar')

            state_dict = checkpoint['state_dict']
            
            model_dict = self.netF.state_dict()
            # 1. filter out unnecessary keys
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(state_dict) 
            # 3. load the new state dict
            self.netF.load_state_dict(model_dict)
            print('resume F network weights successfully')
        else:
            raise Exception("unknown representation")
        
        if self.args.parallel:
            if torch.cuda.device_count()>1:
                self.netG = torch.nn.DataParallel(self.netG, device_ids=[0,1]).cuda()
        

        train_op.parameters_count(self.netG, 'netG')

        # setup optimizer
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # resume if specified
        if self.args.resume: self.load_checkpoint()
        

        useScheduler=False
        if useScheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, [1000,2000], 0.1
            )

        

    def userConfig(self):
        """
        include the task specific setup here
        """
        if self.args.featurelearning:
            assert('f' in self.args.outputType)
        pointer = 0
        if 'rgb' in self.args.outputType:
            self.args.idx_rgb_start = pointer
            self.args.idx_rgb_end = pointer + 3
            pointer += 3
        if 'n' in self.args.outputType:
            self.args.idx_n_start = pointer
            self.args.idx_n_end   = pointer + 3
            pointer += 3
        if 'd' in self.args.outputType:
            self.args.idx_d_start = pointer
            self.args.idx_d_end   = pointer + 1
            pointer += 1
        if 'k' in self.args.outputType:
            self.args.idx_k_start = pointer
            self.args.idx_k_end   = pointer + 1
            pointer += 1
        if 's' in self.args.outputType:
            self.args.idx_s_start = pointer
            self.args.idx_s_end   = pointer + self.args.snumclass # 21 class
            pointer += self.args.snumclass
        if 'f' in self.args.outputType:
            self.args.idx_f_start = pointer
            self.args.idx_f_end   = pointer + self.args.featureDim
            pointer += self.args.featureDim

        self.args.num_output = pointer
        self.args.num_input = 8*2
        self.args.ngpu = int(1)
        self.args.nz = int(100)
        self.args.ngf = int(64)
        self.args.ndf = int(64)
        self.args.nef = int(64)
        self.args.nBottleneck = int(4000)
        self.args.wt_recon = float(0.998)
        self.args.wtlD = float(0.002)
        self.args.overlapL2Weight = 10

        # setup logger
        self.tensorboardX = SummaryWriter(log_dir=os.path.join(self.args.EXP_DIR, 'tensorboard'))
        self.logger = log.logging(self.args.EXP_DIR_LOG)
        self.logger_errG      = AverageMeter()
        self.logger_errG_recon   = AverageMeter()
        self.logger_errG_rgb  = AverageMeter()
        self.logger_errG_d    = AverageMeter()
        self.logger_errG_n    = AverageMeter()
        self.logger_errG_s    = AverageMeter()
        self.logger_errG_k    = AverageMeter()
        
        self.logger_errD_fake = AverageMeter()
        self.logger_errD_real = AverageMeter()
        self.logger_errG_fl   =   AverageMeter()
        self.logger_errG_fl_pos =   AverageMeter()
        self.logger_errG_fl_neg =   AverageMeter()
        self.logger_errG_fl_f =   AverageMeter()
        self.logger_errG_fc   =   AverageMeter()
        self.logger_errG_pn   =   AverageMeter()
        self.logger_errG_freq =   AverageMeter()

        self.global_step=0
        self.speed_benchmark=True
        if self.speed_benchmark:
            self.time_per_step=AverageMeter()

        self.sift = cv2.xfeatures2d.SIFT_create()
        self.evalFeatRatioDL_obs,self.evalFeatRatioDL_unobs=[],[]
        self.evalFeatRatioDLc_obs,self.evalFeatRatioDLc_unobs=[],[]
        self.evalFeatRatioSift=[]
        self.evalErrN=[]
        self.evalErrD=[]
        self.evalSemantic    = []
        self.evalSemantic_gt = []

        self.sancheck={}

        # semantic encoding
        if 'scannet' in self.args.dataList:
            self.colors = config.scannet_color_palette
        elif 'matterport' in self.args.dataList:
            self.colors = config.matterport_color_palette
        elif 'suncg' in self.args.dataList:
            self.colors = config.suncg_color_palette

        self.class_balance_weights = torch_op.v(np.ones([self.args.snumclass]))


    def set_mode(self,mode='train'):
        if mode == 'train':
            self.netG.train()
        else:
            return #!!!
            self.netG.eval()
            return

    def update_lr(self):
        self.lr_scheduler.step()

    def save_checkpoint_helper(self,net,optimizer,filename,clean=True,epoch=None):
        # find previous saved model and only retain the 5 most recent model
        state = {
                'epoch':epoch,
                'state_dict': net.state_dict(),
                'optimizer' : optimizer.state_dict()
            }
        torch.save(state, filename)
        if clean:
            NumRetain=3
            dirname=os.path.dirname(filename)
            checkpointName=filename.split('/')[-1]
            num=re.findall(r'\d+', checkpointName)[0]
            checkpointName=checkpointName.replace(num,'*')
            checkpoints=glob.glob(f"{dirname}/{checkpointName}")
            checkpoints.sort()
            for i in range(len(checkpoints)-NumRetain):
                cmd=f"rm {checkpoints[i]}"
                os.system(cmd)

    def save_checkpoint(self, context):
        epoch = context['epoch']
        self.logger('save model: {0}'.format(epoch))
        self.save_checkpoint_helper(self.netG,self.optimizerG,\
            os.path.join(self.args.EXP_DIR_PARAMS, 'checkpoint_G_{0:04d}.pth.tar'.format(epoch)),clean=True,epoch=epoch)

    def load_checkpoint(self):
        try:
            if self.args.model is not None:
                net_path = self.args.model
            else:
                net_path = train_op.get_latest_model(self.args.EXP_DIR_PARAMS, 'checkpoint')
            
            checkpoint = torch.load(net_path)
            state_dict = checkpoint['state_dict']
            self.epochStart = checkpoint['epoch']+1
            self.netG.load_state_dict(state_dict)

            print('resume network weights from {0} successfully'.format(net_path))
            self.optimizerG.load_state_dict(checkpoint['optimizer'])
            print('resume optimizer weights from {0} successfully'.format(net_path))
        except Exception as e: 
            print(e)
            print("resume fail, start training from scratch!")

    def evalPlot(self,context):
        # normal angle error
        visEvalErrNc=plotCummulative(np.array(self.evalErrN),'angular error','percentage','ours')
        visEvalErrNh=plotHistogram(np.array(self.evalErrN),'angular error','probability','ours')

        # plane distance
        visEvalErrDc=plotCummulative(np.array(self.evalErrD),'l1 error','percentage','ours')
        visEvalErrDh=plotHistogram(np.array(self.evalErrD),'l1 error','probability','ours')

        # semantic class distribution
        if self.args.objectFreqLoss:
            self.evalSemantic = np.concatenate(self.evalSemantic,0).mean(0)
            self.evalSemantic_gt = np.concatenate(self.evalSemantic_gt,0).mean(0)

        visEvalSemantic=plotSeries([range(len(self.colors)),range(len(self.colors))],\
            [self.evalSemantic,self.evalSemantic_gt],'class','pixel percentage',['ours','gt'])

        # descriptive power of learned feature
        visEvalFeat=plotCummulative([np.array(self.evalFeatRatioDLc_obs),np.array(self.evalFeatRatioDLc_unobs),np.array(self.evalFeatRatioDL_obs),np.array(self.evalFeatRatioDL_unobs),np.array(self.evalFeatRatioSift)],'ratio','percentage',['dl_complete_obs','dl_complete_unobs','dl_partial_obs','dl_partial_unobs','sift'])
        visEval = np.concatenate((visEvalErrNc,visEvalErrNh,visEvalErrDc,visEvalErrDh,visEvalFeat,visEvalSemantic),1)

        cv2.imwrite(os.path.join(self.args.EXP_DIR,f"evalMetric_epoch_{context['epoch']}.png"),visEval)

        self.evalErrN=[]
        self.evalErrD=[]
        self.evalSemantic=[]
        self.evalSemantic_gt=[]


    def evalSiftDescriptor(self,rgb,denseCorres):
        ratios=[]
        n=rgb.shape[0]
        Kn = denseCorres['idxSrc'].shape[1]
        for jj in range(n):
            if denseCorres['valid'][jj].item() == 0:
                continue
            idx=np.random.choice(range(Kn),100)
            rs=(torch_op.npy(rgb[jj,0,:,:,:]).transpose(1,2,0)*255).astype('uint8')
            grays= cv2.cvtColor(rs,cv2.COLOR_BGR2GRAY)
            rt=(torch_op.npy(rgb[jj,1,:,:,:]).transpose(1,2,0)*255).astype('uint8')
            grayt= cv2.cvtColor(rt,cv2.COLOR_BGR2GRAY)
            step_size = 5
            tp=torch_op.npy(denseCorres['idxSrc'][jj,idx,:])
            kp = [cv2.KeyPoint(coord[0], coord[1], step_size) for coord in tp]
            _,sifts = self.sift.compute(grays, kp)
            tp=torch_op.npy(denseCorres['idxTgt'][jj,idx,:])
            kp = [cv2.KeyPoint(coord[0], coord[1], step_size) for coord in tp]
            _,siftt = self.sift.compute(grayt, kp)

            dist=np.power(sifts-siftt,2).sum(1)
            
            kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, rgb.shape[3], step_size)
                                                for x in range(0, rgb.shape[4], step_size)]
            _,dense_feat = self.sift.compute(grayt, kp)
            distRest=np.power(np.expand_dims(sifts,1)-np.expand_dims(dense_feat,0),2).sum(2)
            ratio=(distRest<dist[:,np.newaxis]).sum(1)/distRest.shape[1]
            ratios.append(ratio.mean())
        return ratios

    def evalDLDescriptor(self,featMaps,featMapt,denseCorres,rs,rt,mask):
        Kn = denseCorres['idxSrc'].shape[1]
        C  = featMaps.shape[1]
        n  = featMaps.shape[0]
        ratiosObs,ratiosUnobs=[],[]
        rsnpy,rtnpy,masknpy=torch_op.npy(rs),torch_op.npy(rt),torch_op.npy(mask)
        # dim the image to illustrate mask area
        rsnpy = rsnpy * masknpy + 0.5*rsnpy * (1-masknpy)
        rtnpy = rtnpy * masknpy + 0.5*rtnpy * (1-masknpy)

        plot_all=[]
        try:
            for jj in range(n):
                if denseCorres['valid'][jj].item() == 0:
                    continue
                idx=np.random.choice(range(Kn),100)
                
                typeCP  = torch_op.npy(denseCorres['observe'][jj,idx])
                featSrc = featMaps[jj,:,denseCorres['idxSrc'][jj,idx,1].long(),denseCorres['idxSrc'][jj,idx,0].long()]
                featTgt = featMapt[jj,:,denseCorres['idxTgt'][jj,idx,1].long(),denseCorres['idxTgt'][jj,idx,0].long()]
                dist    = (featSrc-featTgt).pow(2).sum(0)
                distRest= (featSrc.unsqueeze(2) - featMapt[jj].view(C,1,-1)).pow(2).sum(0)
                ratio   = (distRest<dist.unsqueeze(1)).sum(1).float()/distRest.shape[1]
                ratio   = torch_op.npy(ratio)
                
                if ((typeCP==2).sum()>0):
                    ratiosObs.append(ratio[typeCP==2].mean())
                if ((typeCP<2).sum()>0):
                    ratiosUnobs.append(ratio[typeCP<2].mean())
        except:
            import ipdb;ipdb.set_trace()
        return ratiosObs,ratiosUnobs,plot_all

    def sancheck_total_traversed(self, imgsPath):
        #print(imgsPath)
        for kk in range(len(imgsPath[0])):
            if imgsPath[0][kk] not in self.sancheck:
                self.sancheck[imgsPath[0][kk]]=1
            else:
                self.sancheck[imgsPath[0][kk]]+=1
        for kk in range(len(imgsPath[1])):
            if imgsPath[1][kk] not in self.sancheck:
                self.sancheck[imgsPath[1][kk]]=1
            else:
                self.sancheck[imgsPath[1][kk]]+=1

    def contrast_loss(self,featMaps,featMapt,denseCorres):
        validCorres=torch.nonzero(denseCorres['valid']==1).view(-1).long()
        n = featMaps.shape[0]
        if not len(validCorres):
            loss_fl_pos=torch_op.v(np.array([0]))[0]
            loss_fl_neg=torch_op.v(np.array([0]))[0]
            loss_fl=torch_op.v(np.array([0]))[0][0]
            loss_fc=torch_op.v(np.array([0]))[0]
        else:
            # consistency of keypoint proposal across different view
            idxInst=torch.arange(n)[validCorres].view(-1,1).repeat(1,denseCorres['idxSrc'].shape[1]).view(-1).long()
            featS=featMaps[idxInst,:,denseCorres['idxSrc'][validCorres,:,1].view(-1).long(),denseCorres['idxSrc'][validCorres,:,0].view(-1).long()]
            featT=featMapt[idxInst,:,denseCorres['idxTgt'][validCorres,:,1].view(-1).long(),denseCorres['idxTgt'][validCorres,:,0].view(-1).long()]
            
            # positive example, 
            loss_fl_pos=(featS-featT).pow(2).sum(1).mean()
            # negative example, make sure does not contain positive
            Kn = denseCorres['idxSrc'].shape[1]
            C = featMaps.shape[1]
            
            negIdy=torch.from_numpy(np.random.choice(range(featMaps.shape[2]),Kn*100*len(validCorres)))
            negIdx=torch.from_numpy(np.random.choice(range(featMaps.shape[3]),Kn*100*len(validCorres)))
            idx=torch.arange(n)[validCorres].view(-1,1).repeat(1,Kn*100).view(-1).long()

            loss_fl_neg=F.relu(self.args.D-(featS.unsqueeze(1).repeat(1,100,1).view(-1,C)-featMapt[idx,:,negIdy,negIdx]).pow(2).sum(1)).mean()
            loss_fl=loss_fl_pos+loss_fl_neg
            return loss_fl, loss_fl_pos, loss_fl_neg

    def step(self,data,mode='train'):
        torch.cuda.empty_cache()
        if self.speed_benchmark:
            step_start=time.time()
        
        with torch.set_grad_enabled(mode == 'train'):
            np.random.seed()
            self.optimizerG.zero_grad()
            
            MSEcriterion = torch.nn.MSELoss()
            BCEcriterion = torch.nn.BCELoss()
            CEcriterion  = nn.CrossEntropyLoss(weight=self.class_balance_weights,reduce=False)
            
            rgb,norm,depth,dataMask,Q = v(data['rgb']),v(data['norm']),v(data['depth']),v(data['dataMask']),v(data['Q'])
            proj_rgb_p,proj_n_p,proj_d_p,proj_mask_p = v(data['proj_rgb_p']),v(data['proj_n_p']),v(data['proj_d_p']),v(data['proj_mask_p'])
            proj_flow = v(data['proj_flow'])
            if 's' in self.args.outputType: segm    = v(data['segm'])
            if self.args.dynamicWeighting:  
                dynamicW = v(data['proj_box_p'])
                dynamicW[dynamicW==0] = 0.2
                dynamicW = torch.cat((dynamicW[:,0,:,:,:],dynamicW[:,1,:,:,:]))
            else:
                dynamicW = 1
            errG_rgb,errG_d,errG_n,errG_k,errG_s = torch.FloatTensor([0]),torch.FloatTensor([0]),torch.FloatTensor([0]),torch.FloatTensor([0]),torch.FloatTensor([0])

            n = Q.shape[0]

            complete_s=torch.cat((rgb[:,0,:,:,:],norm[:,0,:,:,:],depth[:,0:1,:,:]),1)
            complete_t=torch.cat((rgb[:,1,:,:,:],norm[:,1,:,:,:],depth[:,1:2,:,:]),1)
            
            view_s,mask_s,geow_s = apply_mask(complete_s.clone(),self.args.maskMethod,self.args.ObserveRatio)
            view_s = torch.cat((view_s,mask_s),1)
            view_t,mask_t,geow_t = apply_mask(complete_t.clone(),self.args.maskMethod,self.args.ObserveRatio)
            view_t = torch.cat((view_t,mask_t),1)

            view_t2s=torch.cat((proj_rgb_p[:,0,:,:,:],proj_n_p[:,0,:,:,:],proj_d_p[:,0,:,:,:],proj_mask_p[:,0,:,:,:]),1)
            view_s2t=torch.cat((proj_rgb_p[:,1,:,:,:],proj_n_p[:,1,:,:,:],proj_d_p[:,1,:,:,:],proj_mask_p[:,1,:,:,:]),1)


            # netG need to tolerate three type of input: 
            # 0.correct s + blank t 
            # 1.correct s + wrong t 
            # 2.correct s + correct t
            view_s_type0 = torch.cat((view_s,torch.zeros(view_s.shape).cuda()),1)
            view_s_type1 = torch.cat((view_s,view_t2s),1)

            view_t_type0 = torch.cat((view_t,torch.zeros(view_t.shape).cuda()),1)
            view_t_type1 = torch.cat((view_t,view_s2t),1)

            if 's' in self.args.outputType:
                segm = torch.cat((segm[:,0,:,:,:],segm[:,1,:,:,:])).repeat(2,1,1,1)

            # mask the pano
            view=torch.cat((view_s_type0,view_t_type0,view_s_type1,view_t_type1))
            mask=torch.cat((mask_s,mask_t)).repeat(2,1,1,1)
            geow=torch.cat((geow_s,geow_t)).repeat(2,1,1,1)
            complete =torch.cat((complete_s,complete_t)).repeat(2,1,1,1)
            dataMask = torch.cat((dataMask[:,0,:,:,:],dataMask[:,1,:,:,:])).repeat(2,1,1,1)
            
            fake      = self.netG(view)
            with torch.set_grad_enabled(False):
                fakec     = self.netF(complete)
            if 'f' in self.args.outputType:
                featMapsc = fakec[:n]
                featMaptc = fakec[n:n*2]
                if np.random.rand()>0.5:
                    featMaps  = fake[:n,self.args.idx_f_start:self.args.idx_f_end,:,:]
                    featMapt  = fake[n:n*2,self.args.idx_f_start:self.args.idx_f_end,:,:]
                else:
                    featMaps  = fake[n*2:n*3,self.args.idx_f_start:self.args.idx_f_end,:,:]
                    featMapt  = fake[n*3:n*4,self.args.idx_f_start:self.args.idx_f_end,:,:]

            
            if self.args.featurelearning:
                denseCorres = data['denseCorres']
                validCorres=torch.nonzero(denseCorres['valid']==1).view(-1).long()
                loss_fl, loss_fl_pos, loss_fl_neg = self.contrast_loss(featMaps,featMapt,data['denseCorres'])

                # categorize each correspondence by whether it contain unobserved point
                allCorres = torch.cat((denseCorres['idxSrc'],denseCorres['idxTgt']))
                corresShape = allCorres.shape
                allCorres = allCorres.view(-1,2).long()
                typeIdx = torch.arange(corresShape[0]).view(-1,1).repeat(1,corresShape[1]).view(-1).long()
                typeIcorresP = mask[typeIdx,0,allCorres[:,1],allCorres[:,0]]
                typeIcorresP=typeIcorresP.view(2,-1,corresShape[1]).sum(0)
                denseCorres['observe'] = typeIcorresP
                
                loss_fc=torch.pow((fake[:,self.args.idx_f_start:self.args.idx_f_end,:,:]-fakec.detach())*dataMask*geow,2).sum(1).mean()


            errG_recon = 0
        
            if self.args.GeometricWeight:
                total_weight = geow[:,0:1,:,:]*dynamicW*dataMask
            else:
                total_weight = dynamicW*dataMask
            if 'rgb' in self.args.outputType:
                errG_rgb = ((fake[:,self.args.idx_rgb_start:self.args.idx_rgb_end,:,:]-complete[:,0:3,:,:])*total_weight).abs().mean()
                errG_recon += errG_rgb
            if 'n' in self.args.outputType:
                errG_n   = ((fake[:,self.args.idx_n_start:self.args.idx_n_end,:,:]-complete[:,3:6,:,:])*total_weight).abs().mean()
                errG_recon += errG_n 
            if 'd' in self.args.outputType:
                errG_d   = ((fake[:,self.args.idx_d_start:self.args.idx_d_end,:,:]-complete[:,6:7,:,:])*total_weight).abs().mean()
                errG_recon += errG_d
            if 'k' in self.args.outputType:
                errG_k   = ((fake[:,self.args.idx_k_start:self.args.idx_k_end,:,:]-complete[:,7:8,:,:])*total_weight).abs().mean()
                errG_recon += errG_k
            if 's' in self.args.outputType:
                errG_s   = (CEcriterion(fake[:,self.args.idx_s_start:self.args.idx_s_end,:,:],segm.squeeze(1).long())*total_weight).mean() * 0.1
                errG_recon += errG_s
        
            
            errG = errG_recon

            if self.args.pnloss:
                loss_pn = util.pnlayer(torch.cat((depth[:,0:1,:,:],depth[:,1:2,:,:])),fake[:,3:6,:,:],fake[:,6:7,:,:]*4,self.args.dataList,self.args.representation)*1e-1
                #loss_pn = util.pnlayer(torch.cat((depth[:,0:1,:,:],depth[:,1:2,:,:])),complete[:,3:6,:,:],complete[:,6:7,:,:]*4,self.args.dataList,self.args.representation)*1e-1
                errG += loss_pn

            if self.args.featurelearning:
                errG += loss_fl+loss_fc

            #if errG.item()>100:
            #    import ipdb;ipdb.set_trace()
            
            if mode == 'train':
                errG.backward()
                self.optimizerG.step()
            
            self.logger_errG.update(errG.data, Q.size(0))
            self.logger_errG_rgb.update(errG_rgb.data, Q.size(0))
            self.logger_errG_n.update(errG_n.data, Q.size(0))
            self.logger_errG_d.update(errG_d.data, Q.size(0))
            self.logger_errG_s.update(errG_s.data, Q.size(0))
            self.logger_errG_k.update(errG_k.data, Q.size(0))

            if self.args.pnloss:
                self.logger_errG_pn.update(loss_pn.data, Q.size(0))
            if self.args.featurelearning:
                self.logger_errG_fl.update(loss_fl.data, Q.size(0))
                self.logger_errG_fl_pos.update(loss_fl_pos.data, Q.size(0))
                self.logger_errG_fl_neg.update(loss_fl_neg.data, Q.size(0))
                self.logger_errG_fc.update(loss_fc.data, Q.size(0))
            if self.args.objectFreqLoss:
                self.logger_errG_freq.update(loss_freq.data, Q.size(0))

            
            
            suffix = f"| errG {self.logger_errG.avg:.6f}| | errG_fl {self.logger_errG_fl.avg:.6f}\
                 | errG_fl_pos {self.logger_errG_fl_pos.avg:.6f} | errG_fl_neg {self.logger_errG_fl_neg.avg:.6f} | errG_fc {self.logger_errG_fc.avg:.6f} | errG_pn {self.logger_errG_pn.avg:.6f} | errG_freq {self.logger_errG_freq.avg:.6f}"
            
            if self.global_step % getattr(self.learnerParam,f"{mode}_step_vis") == 0:
                print(f"total image trasversed:{len(self.sancheck)}\n")
                # do logging and visualizing

                if 'n' in self.args.outputType:
                    # normalized normal
                    faken = fake[:,self.args.idx_n_start:self.args.idx_n_end,:,:]
                    faken = faken/torch.norm(faken,dim=1,keepdim=True)

                vis = []
                if 'rgb' in self.args.outputType:
                    # draw rgb
                    visrgb  = complete[:,0:3,:,:]
                    visrgbm = view[:,0:3,:,:]
                    visrgbm2 = view[:,8+0:8+3,:,:]
                    visrgbf = fake[:,self.args.idx_rgb_start:self.args.idx_rgb_end,:,:]
                    visrgbf  = visNorm(visrgbf)
                    visrgbc = (fake[:,self.args.idx_rgb_start:self.args.idx_rgb_end,:,:]*(1-mask)+visrgb*mask)
                    visrgbc  = visNorm(visrgbc)
                    visrgb  = torch.cat((visrgbm,visrgbm2,visrgbf,visrgbc,visrgb),2)
                    visrgb  = visNorm(visrgb)
                    vis.append(visrgb)

                if 'n' in self.args.outputType:
                    # draw normal 
                    visn  = complete[:,3:6,:,:]
                    visnm = view[:,3:6,:,:]
                    visnm2 = view[:,8+3:8+6,:,:]
                    visnf = faken
                    visnc = (faken*(1-mask)+visn*mask)
                    visn  = torch.cat((visnm,visnm2,visnf,visnc,visn),2)
                    visn  = visNorm(visn)
                    vis.append(visn)

                if 'd' in self.args.outputType:
                    # draw depth
                    visd  = complete[:,6:7,:,:]
                    visdm = view[:,6:7,:,:]
                    visdm2 = view[:,8+6:8+7,:,:]
                    visdf = fake[:,self.args.idx_d_start:self.args.idx_d_end,:,:]
                    visdc = (fake[:,self.args.idx_d_start:self.args.idx_d_end,:,:]*(1-mask)+visd*mask)
                    visd  = torch.cat((visdm,visdm2,visdf,visdc,visd),2)
                    visd  = visNorm(visd)
                    visd  = visd.repeat(1,3,1,1)
                    vis.append(visd)

                if 'k' in self.args.outputType:
                    # draw keypoint 
                    visk  = complete[:,7:8,:,:]
                    viskm = view[:,7:8,:,:]
                    viskm2 = view[:,8+7:8+8,:,:]
                    viskf = fake[:,self.args.idx_k_start:self.args.idx_k_end:,:]
                    viskc = fake[:,self.args.idx_k_start:self.args.idx_k_end:,:].clone()
                    viskc = viskc*(1-mask)+(viskc.view(viskc.shape[0],-1).min(1)[0].view(-1,1,1,1))*mask
                    viskc = visNorm(viskc)
                    viskc = util.extractKeypoint(viskc)
                    viskc = (viskc*(1-mask)+visk*mask)
                    visk  = torch.cat((viskm,viskf,viskc,visk),2)
                    visk  = visk.repeat(1,3,1,1)
                    vis.append(visk)

                if 's' in self.args.outputType:
                    # draw semantic
                    viss  = segm
                    vissm = viss*mask[:,0:1,:,:]
                    vissf = fake[:,self.args.idx_s_start:self.args.idx_s_end,:,:]
                    vissf = torch.argmax(vissf,1,keepdim=True).float()
                    vissc = (vissf*(1-mask)+viss*mask)
                    viss  = torch.cat((vissm,vissf,vissc,viss),2)
                    visstp= torch_op.npy(viss)
                    visstp= np.expand_dims(np.squeeze(visstp,1),3)
                    visstp= self.colors[visstp.flatten().astype('int'),:].reshape(visstp.shape[0],visstp.shape[1],visstp.shape[2],3)
                    viss  = torch_op.v(visstp.transpose(0,3,1,2))/255.
                    vis.append(viss)

                if self.args.dynamicWeighting:
                    visdw = dynamicW.repeat(1,3,1,1)
                    vis.append(visdw)

                if 'f' in self.args.outputType:
                    # draw feature error map
                    visf = fake[:,self.args.idx_f_start:self.args.idx_f_end,:,:]
                    visf = (visf - fakec).pow(2).sum(1,keepdim=True)
                    visf = visNorm(visf)
                    visf = visf.repeat(1,3,1,1)
                    vis.append(visf)
                    

                visw = total_weight.repeat(1,3,1,1)
                vis.append(visw)

                # concate all vis
                vis = torch.cat(vis, 2)[::2]
                permute = [2, 1, 0] # bgr to rgb
                vis   = vis[:,permute,:,:]

                if mode != 'train':
                    with torch.set_grad_enabled(False):
                        if 'n' and 'd' in self.args.outputType:
                            # evaluate strcuture prediction
                            
                            ## 1. normal angle
                            mask_n=(1-mask[:,0:1,:,:]).cpu()
                            mask_n = mask_n * dataMask.cpu()
                            
                            evalErrN=(torch.acos(((faken.cpu()*complete[:,3:6,:,:].cpu()).sum(1,keepdim=True)[mask_n!=0]).clamp(-1,1))/np.pi*180)
                            self.evalErrN.extend(npy(evalErrN))

                            ## 2. plane distance
                            evalErrD=((fake[:,6:7,:,:].cpu()-complete[:,6:7,:,:].cpu())[mask_n!=0]).abs()
                            self.evalErrD.extend(npy(evalErrD))

                        # evaluate the learned feature
                        ## 1. descriptive power of learned feature
                        if self.args.featurelearning:
                            if len(validCorres):
                                
                                self.evalFeatRatioSift.extend(self.evalSiftDescriptor(rgb,denseCorres))
                                obs,unobs,_=self.evalDLDescriptor(featMapsc,featMaptc,denseCorres,complete_s[:,0:3,:,:],complete_t[:,0:3,:,:],mask[0:1,0:1,:,:])
                                self.evalFeatRatioDLc_obs.extend(obs)
                                self.evalFeatRatioDLc_unobs.extend(unobs)
                                obs,unobs,_=self.evalDLDescriptor(featMaps,featMapt,denseCorres,complete_s[:,0:3,:,:],complete_t[:,0:3,:,:],mask[0:1,0:1,:,:])

                                self.evalFeatRatioDL_obs.extend(obs)
                                self.evalFeatRatioDL_unobs.extend(unobs)
                        
                        if self.args.objectFreqLoss:
                            freq_pred = freq_pred/freq_pred.sum(1,keepdim=True)
                            freq_gt   = freq_gt/freq_gt.sum(1,keepdim=True)
                            self.evalSemantic.append(torch_op.npy(freq_pred))
                            self.evalSemantic_gt.append(torch_op.npy(freq_gt))

                train_op.tboard_add_img(self.tensorboardX,vis,f"{mode}/loss",self.global_step)

            if self.global_step % getattr(self.learnerParam,f"{mode}_step_log") == 0:
                self.tensorboardX.add_scalars('data/errG_recon', {f"{mode}":errG_recon}, self.global_step)
                self.tensorboardX.add_scalars('data/errG_rgb', {f"{mode}":errG_rgb}, self.global_step)
                self.tensorboardX.add_scalars('data/errG_n', {f"{mode}":errG_n}, self.global_step)
                self.tensorboardX.add_scalars('data/errG_d', {f"{mode}":errG_d}, self.global_step)
                self.tensorboardX.add_scalars('data/errG_s', {f"{mode}":errG_s}, self.global_step)
                self.tensorboardX.add_scalars('data/errG_k', {f"{mode}":errG_k}, self.global_step)
                if self.args.pnloss:
                    self.tensorboardX.add_scalars('data/errG_pnloss', {f"{mode}":loss_pn}, self.global_step)
                if self.args.featurelearning:
                    self.tensorboardX.add_scalars('data/errG_fl', {f"{mode}_complete":loss_fl}, self.global_step)
                    self.tensorboardX.add_scalars('data/errG_fl_pos', {f"{mode}_complete":loss_fl_pos}, self.global_step)
                    self.tensorboardX.add_scalars('data/errG_fl_neg', {f"{mode}_complete":loss_fl_neg}, self.global_step)
                    self.tensorboardX.add_scalars('data/errG_fc', {f"{mode}":loss_fc}, self.global_step)
                if self.args.objectFreqLoss:
                    self.tensorboardX.add_scalars('data/errG_freq', {f"{mode}":loss_freq}, self.global_step)
            
            summary = {'suffix':suffix}
            self.global_step+=1
        
        if self.speed_benchmark:
            self.time_per_step.update(time.time()-step_start,1)
            print(f"time elapse per step: {self.time_per_step.avg}")
        return dotdict(summary)

def main():
    # parse arguments, build exp dir
    opt = opts()
    args = opt.parse()
    train_op.initialize_experiment_directories(args)
    train_op.platform_specific_initialization(args)

    # build data loader
    train_loader,val_loader=buildDataset(args)

    # build learner
    lp = learnerParam()
    model=learner(args,lp)
    
    # build trainer and launch training
    mytrainer=trainer(
        model,
        train_loader,
        val_loader,
        max_epoch=200,
        )

    mytrainer.add_callbacks([PeriodicCallback(cb_loc=CallbackLoc.epoch_end,pstep=5,func=model.save_checkpoint)])
    mytrainer.add_callbacks([PeriodicCallback(cb_loc=CallbackLoc.epoch_end,pstep=5,func=model.evalPlot)])

    mytrainer.run()

if __name__ == '__main__':
    main()
