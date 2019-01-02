import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from util import angular_distance_np
from utils.plot import plotCummulative
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
from model.context_encoder import Resnet18_8s,segmentation_layer
from utils.callbacks import PeriodicCallback, OnceCallback, ScheduledCallback,CallbackLoc
from sklearn.decomposition import PCA

#**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--
#**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--**--*--
# here is the place for customized functions

def visNorm(vis):
    for v in range(len(vis)):
        if (vis[v].max().item() - vis[v].min().item())!=0:
            vis[v] = (vis[v]-vis[v].min())/(vis[v].max()-vis[v].min())
    return vis

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

    train_dataset = Dataset('train', config.nViews,AuthenticdepthMap=False,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True
    ,list_=f"./data/dataList/{args.dataList}.npy",singleView=args.single_view,denseCorres=True,reproj=False,representation=args.representation)
    val_dataset = Dataset('test', nViews=config.nViews,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,\
        list_=f"./data/dataList/{args.dataList}.npy",singleView=args.single_view,denseCorres=True,reproj=False,representation=args.representation)

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
            self.netF=Resnet18_8s(args).cuda()
        
        self.netSemg = segmentation_layer(args).cuda()

        train_op.parameters_count(self.netF, 'netF')

        # setup optimizer
        
        params = list(self.netF.parameters()) + list(self.netSemg.parameters())
        self.optimizerF = torch.optim.Adam(params, lr=0.0002, betas=(0.5, 0.999))

        # resume if specified
        if self.args.resume: self.load_checkpoint()

    def userConfig(self):
        """
        include the task specific setup here
        """
        if self.args.featurelearning:
            self.args.outputType += 'f'
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
            self.args.idx_f_end   = pointer + 32
            pointer += 32

        self.args.num_output = pointer
        self.args.num_input = 7
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
            self.netF.train()
        else:
            return #
            self.netF.eval()
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
        if self.args.ganloss:
            self.save_checkpoint_helper(self.netD,self.optimizerD,\
                os.path.join(self.args.EXP_DIR_PARAMS, 'checkpoint_D_{0:04d}.pth.tar'.format(epoch)),clean=True,epoch=epoch)
        self.save_checkpoint_helper(self.netF,self.optimizerF,\
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

            model_dict = self.netF.state_dict()
            # 1. filter out unnecessary keys
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(state_dict) 
            # 3. load the new state dict
            self.netF.load_state_dict(model_dict)

            #self.netF.load_state_dict(state_dict)
            print('resume network weights from {0} successfully'.format(net_path))
            self.optimizerF.load_state_dict(checkpoint['optimizer'])
            print('resume optimizer weights from {0} successfully'.format(net_path))
        except:
            print("resume fail, start training from scratch!")

    def evalPlot(self,context):
        # descriptive power of learned feature
        visEvalFeat=plotCummulative([np.array(self.evalFeatRatioDLc_obs),np.array(self.evalFeatRatioDLc_unobs),np.array(self.evalFeatRatioSift)],'ratio','percentage',['dl_complete_obs','dl_complete_unobs','sift'])
        cv2.imwrite(os.path.join(self.args.EXP_DIR,f"evalMetric_epoch_{context['epoch']}.png"),visEvalFeat)
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
        for jj in range(n):
            if denseCorres['valid'][jj].item() == 0:
                continue
            idx=np.random.choice(range(Kn),100)
            if self.args.debug:
                tp=util.drawMatch(rsnpy[jj,:,:,:].transpose(1,2,0)*255,rtnpy[jj,:,:,:].transpose(1,2,0)*255,torch_op.npy(denseCorres['idxSrc'][jj,idx,:]),torch_op.npy(denseCorres['idxTgt'][jj,idx,:]))
                cv2.imwrite('Debug_evalDLDescriptor_0.png',tp)
            
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

            ratioIdx = np.argsort(ratio)
            # plot 10 most confident correspondence
            plot_this = []
            for kk in range(10):
                tp=util.drawMatch(rsnpy[jj,:,:,:].transpose(1,2,0)*255,rtnpy[jj,:,:,:].transpose(1,2,0)*255,\
                    torch_op.npy(denseCorres['idxSrc'][jj,[idx[ratioIdx[kk]]],:]),torch_op.npy(denseCorres['idxTgt'][jj,[idx[ratioIdx[kk]]],:]))
                probMap=torch_op.npy(distRest[ratioIdx[kk],:].reshape(featMaps.shape[2],featMaps.shape[3]))
                sigmap=np.median(probMap)/4
                probMapL1=np.exp(-probMap/(2*sigmap**2))
                probMapL1=np.tile(np.expand_dims((probMapL1-probMapL1.min())/(probMapL1.max()-probMapL1.min()),2),[1,1,3])
                # overlay probability map to rgb
                tp1=(rtnpy[jj,:,:,:].transpose(1,2,0)*(probMapL1!=0)+np.array([0,1,0]).reshape(1,1,3)*probMapL1)*255
                tp=np.concatenate((tp,tp1))
                probMapL2=np.exp(-probMap/(2*(sigmap)**2))
                probMapL2=np.tile(np.expand_dims((probMapL2-probMapL2.min())/(probMapL2.max()-probMapL2.min())*255,2),[1,1,3])
                tp=np.concatenate((tp,probMapL2))
                plot_this.append(tp)
                plot_this.append(np.zeros([tp.shape[0],20,3]))
            plot_this=np.concatenate(plot_this,1)
            plot_all.append(plot_this)
            
        return ratiosObs,ratiosUnobs,plot_all



    def step(self,data,mode='train'):
        torch.cuda.empty_cache()
        if self.speed_benchmark:
            step_start=time.time()
        with torch.set_grad_enabled(mode == 'train'):
            np.random.seed()
            self.optimizerF.zero_grad()
            
            MSEcriterion = torch.nn.MSELoss()
            BCEcriterion = torch.nn.BCELoss()
            CEcriterion  = nn.CrossEntropyLoss(weight=self.class_balance_weights,reduce=False)
            
            rgb,norm,depth,dataMask,Q = v(data['rgb']),v(data['norm']),v(data['depth']),v(data['dataMask']),v(data['Q'])
            segm    = v(data['segm'])
            segm = torch.cat((segm[:,0,:,:,:],segm[:,1,:,:,:]))
            errG_rgb,errG_d,errG_n,errG_k,errG_s = torch.FloatTensor([0]),torch.FloatTensor([0]),torch.FloatTensor([0]),torch.FloatTensor([0]),torch.FloatTensor([0])

            n = Q.shape[0]
            
            # compose the input: [rgb, normal, depth]
            complete0=torch.cat((rgb[:,0,:,:,:],norm[:,0,:,:,:],depth[:,0:1,:,:]),1)
            complete1=torch.cat((rgb[:,1,:,:,:],norm[:,1,:,:,:],depth[:,1:2,:,:]),1)
            
            view0,mask0,geow0 = apply_mask(complete0.clone(),self.args.maskMethod,self.args.ObserveRatio)
            view1,mask1,geow1 = apply_mask(complete1.clone(),self.args.maskMethod,self.args.ObserveRatio)
            view=torch.cat((view0,view1))
            mask=torch.cat((mask0,mask1))

            # mask the pano
            complete =torch.cat((complete0,complete1))
            dataMask = torch.cat((dataMask[:,0,:,:,:],dataMask[:,1,:,:,:]))
            
            fakec     = self.netF(complete)
            segm_pred = self.netSemg(fakec)

            featMapsc = fakec[:n]
            featMaptc = fakec[n:]
            
            denseCorres = data['denseCorres']
            validCorres=torch.nonzero(denseCorres['valid']==1).view(-1).long()
            
            if not len(validCorres):
                loss_fl_pos=torch_op.v(np.array([0]))[0]
                loss_fl_neg=torch_op.v(np.array([0]))[0]
                loss_fl=torch_op.v(np.array([0]))[0][0]
                loss_fc=torch_op.v(np.array([0]))[0]
                loss_fl_pos_f=torch_op.v(np.array([0]))[0]
                loss_fl_neg_f=torch_op.v(np.array([0]))[0]
                loss_fl_f=torch_op.v(np.array([0]))[0]
            else:
                # categorize each correspondence by whether it contain unobserved point
                allCorres = torch.cat((denseCorres['idxSrc'],denseCorres['idxTgt']))
                corresShape = allCorres.shape
                allCorres = allCorres.view(-1,2).long()
                typeIdx = torch.arange(corresShape[0]).view(-1,1).repeat(1,corresShape[1]).view(-1).long()
                typeIcorresP = mask[typeIdx,0,allCorres[:,1],allCorres[:,0]]
                typeIcorresP=typeIcorresP.view(2,-1,corresShape[1]).sum(0)
                denseCorres['observe'] = typeIcorresP

                # consistency of keypoint proposal across different view
                idxInst=torch.arange(n)[validCorres].view(-1,1).repeat(1,denseCorres['idxSrc'].shape[1]).view(-1).long()
                featS=featMapsc[idxInst,:,denseCorres['idxSrc'][validCorres,:,1].view(-1).long(),denseCorres['idxSrc'][validCorres,:,0].view(-1).long()]
                featT=featMaptc[idxInst,:,denseCorres['idxTgt'][validCorres,:,1].view(-1).long(),denseCorres['idxTgt'][validCorres,:,0].view(-1).long()]
                
                # positive example, 
                loss_fl_pos=(featS-featT).pow(2).sum(1).mean()
                
                # negative example, make sure does not contain positive
                Kn = denseCorres['idxSrc'].shape[1]
                C = featMapsc.shape[1]
                
                negIdy=torch.from_numpy(np.random.choice(range(featMapsc.shape[2]),Kn*100*len(validCorres)))
                negIdx=torch.from_numpy(np.random.choice(range(featMapsc.shape[3]),Kn*100*len(validCorres)))
                idx=torch.arange(n)[validCorres].view(-1,1).repeat(1,Kn*100).view(-1).long()
                
                loss_fl_neg=F.relu(self.args.D-(featS.unsqueeze(1).repeat(1,100,1).view(-1,C)-featMaptc[idx,:,negIdy,negIdx]).pow(2).sum(1)).mean()
                loss_fl=loss_fl_pos+loss_fl_neg

            
            errG = loss_fl
            total_weight = dataMask
            if self.args.featlearnSegm:
                errG_s   = (CEcriterion(segm_pred,segm.squeeze(1).long())*total_weight).mean() * 0.1
                errG += errG_s

            
            if mode == 'train' and len(validCorres) > 0:
                errG.backward()
                self.optimizerF.step()
            
            self.logger_errG.update(errG.data, Q.size(0))

            self.logger_errG_fl.update(loss_fl.data, Q.size(0))
            self.logger_errG_fl_pos.update(loss_fl_pos.data, Q.size(0))
            self.logger_errG_fl_neg.update(loss_fl_neg.data, Q.size(0))
            
            
            suffix = f"| errG {self.logger_errG.avg:.6f}| errG_fl {self.logger_errG_fl.avg:.6f}\
                 | errG_fl_pos {self.logger_errG_fl_pos.avg:.6f} | errG_fl_neg {self.logger_errG_fl_neg.avg:.6f} | errG_fc {self.logger_errG_fc.avg:.6f} | errG_pn {self.logger_errG_pn.avg:.6f} | errG_freq {self.logger_errG_freq.avg:.6f}"
            
            if self.global_step % getattr(self.learnerParam,f"{mode}_step_vis") == 0:
                if mode != 'train':
                    with torch.set_grad_enabled(False):
                        
                        if len(validCorres):
                            self.evalFeatRatioSift.extend(self.evalSiftDescriptor(rgb,denseCorres))
                            obs,unobs,visCPc=self.evalDLDescriptor(featMapsc,featMaptc,denseCorres,complete0[:,0:3,:,:],complete1[:,0:3,:,:],mask[0:1,0:1,:,:])
                            self.evalFeatRatioDLc_obs.extend(obs)
                            self.evalFeatRatioDLc_unobs.extend(unobs)
                            visCPdir=os.path.join(self.args.EXP_DIR_SAMPLES,f"step_{self.global_step}")
                            if not os.path.exists(visCPdir):os.mkdir(visCPdir)
                            for ii in range(len(visCPc)):
                                cv2.imwrite(os.path.join(visCPdir,f"complete_{ii}.png"),visCPc[ii])
                vis=[]
                # draw semantic
                viss  = segm
                vissf = segm_pred
                vissf = torch.argmax(vissf,1,keepdim=True).float()
                viss  = torch.cat((vissf,viss),2)
                visstp= torch_op.npy(viss)
                visstp= np.expand_dims(np.squeeze(visstp,1),3)
                visstp= self.colors[visstp.flatten().astype('int'),:].reshape(visstp.shape[0],visstp.shape[1],visstp.shape[2],3)
                viss  = torch_op.v(visstp.transpose(0,3,1,2))/255.
                vis.append(viss)
                vis = torch.cat(vis, 2)[::2]
                permute = [2, 1, 0] # bgr to rgb
                vis   = vis[:,permute,:,:]
                train_op.tboard_add_img(self.tensorboardX,vis,f"{mode}/loss",self.global_step)

            if self.global_step % getattr(self.learnerParam,f"{mode}_step_log") == 0:
                self.tensorboardX.add_scalars('data/errG_fl', {f"{mode}_complete":loss_fl}, self.global_step)
                self.tensorboardX.add_scalars('data/errG_fl_pos', {f"{mode}_complete":loss_fl_pos}, self.global_step)
                self.tensorboardX.add_scalars('data/errG_fl_neg', {f"{mode}_complete":loss_fl_neg}, self.global_step)
                self.tensorboardX.add_scalars('data/errG_s', {f"{mode}":errG_s}, self.global_step)
                
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
