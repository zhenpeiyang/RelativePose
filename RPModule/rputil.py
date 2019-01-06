
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../')
import util
from utils import torch_op

class opts():
    def __init__(self,sigmaAngle1=0.523/2,sigmaAngle2=0.523/2,sigmaDist=0.08/2,sigmaFeat=0.01):
        self.distThre = 0.08
        self.distSepThre = 1.5*0.08
        self.angleThre = 45/180.*np.pi
        self.sigmaAngle1=sigmaAngle1
        self.sigmaAngle2=sigmaAngle2
        self.sigmaDist=sigmaDist
        self.sigmaFeat=sigmaFeat
        self.mu = 0.3
        self.topK = 5
        self.method = 'irls+sm'

def angular_distance_np(R_hat, R):
    # measure the angular distance between two rotation matrice
    # R1,R2: [n, 3, 3]
    if len(R_hat.shape)==2:
        R_hat=R_hat[np.newaxis,:]
    if len(R.shape)==2:
        R=R[np.newaxis,:]
    n = R.shape[0]
    trace_idx = [0,4,8]
    trace = np.matmul(R_hat, R.transpose(0,2,1)).reshape(n,-1)[:,trace_idx].sum(1)
    metric = np.arccos(((trace - 1)/2).clip(-1,1)) / np.pi * 180.0
    return metric

def visNorm(vis):
    for v in range(len(vis)):
        if (vis[v].max().item() - vis[v].min().item())!=0:
            vis[v] = (vis[v]-vis[v].min())/(vis[v].max()-vis[v].min())
    return vis

def interpolate(feat, pt):
    # feat: c,h,w
    # pt: K,2
    # return: c,k
    h,w = feat.shape[1],feat.shape[2]
    x = pt[:,0]*(w-1)
    y = pt[:,1]*(h-1)
    x0 = torch.floor(x)
    y0 = torch.floor(y)

    val=feat[:,y0.long(),x0.long()]*(x0+1-x)*(y0+1-y)+\
            feat[:,y0.long()+1,x0.long()]*(x0+1-x)*(y-y0)+\
            feat[:,y0.long(),x0.long()+1]*(x-x0)*(y0+1-y)+\
            feat[:,y0.long()+1,x0.long()+1]*(x-x0)*(y-y0)
    
    return val


def getPixel_helper(depth,xs,ys,val,dataset='suncg',representation='skybox'):
    assert(representation == 'skybox')
    assert(depth.shape[0]==160 and depth.shape[1]==640)
    w,h=160,160
    
    Rs = np.zeros([4,4,4])
    Rs[0] = np.eye(4)
    Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
    
    pc = []
    for i in range(len(xs)):
        idx = int(xs[i]//160)
        if 'suncg' in dataset:
            R_this=Rs[idx]
        elif 'scannet' in dataset or 'matterport' in dataset:
            R_this=Rs[(idx-1)%4]
        ystp, xstp = (0.5-ys[i] / h)*2, ((xs[i]-idx*160) / w-0.5)*2
        zstp = val[i]
        ystp, xstp = ystp*zstp, xstp*zstp
        tmp = np.concatenate(([xstp],[ystp],[-zstp]))
        tmp = np.matmul(R_this[:3,:3],tmp)+R_this[:3,3]
        pc.append(tmp)
    pc = np.concatenate(pc).reshape(-1,3)
    return pc

def getPixel(depth, normal, pts, dataset='suncg',representation='skybox'):
    # pts: [n, 2]
    # depth: [dim, dim]
    tp = np.floor(pts).astype('int')
    v1 = depth[tp[:,1],tp[:,0]]
    v2 = depth[tp[:,1],tp[:,0]+1]
    v3 = depth[tp[:,1]+1,tp[:,0]]
    v4 = depth[tp[:,1]+1,tp[:,0]+1]
    val = v1*(tp[:,1]+1-pts[:,1])*(tp[:,0]+1-pts[:,0]) + \
        v2*(pts[:,0]-tp[:,0])*(tp[:,1]+1-pts[:,1]) + \
        v3*(pts[:,1]-tp[:,1])*(tp[:,0]+1-pts[:,0]) + \
        v4*(pts[:,0]-tp[:,0])*(pts[:,1]-tp[:,1])
    
    Rs = np.zeros([4,4,4])
    Rs[0] = np.eye(4)
    Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
    v1 = normal[tp[:,1],tp[:,0],:]
    v2 = normal[tp[:,1],tp[:,0]+1,:]
    v3 = normal[tp[:,1]+1,tp[:,0],:]
    v4 = normal[tp[:,1]+1,tp[:,0]+1,:]
    nn = v1*(tp[:,1]+1-pts[:,1])[:,np.newaxis]*(tp[:,0]+1-pts[:,0])[:,np.newaxis] + \
        v2*(pts[:,0]-tp[:,0])[:,np.newaxis]*(tp[:,1]+1-pts[:,1])[:,np.newaxis] + \
        v3*(pts[:,1]-tp[:,1])[:,np.newaxis]*(tp[:,0]+1-pts[:,0])[:,np.newaxis] + \
        v4*(pts[:,0]-tp[:,0])[:,np.newaxis]*(pts[:,1]-tp[:,1])[:,np.newaxis]
    nn /= np.linalg.norm(nn,axis=1,keepdims=True)
        
    ys, xs = pts[:,1],pts[:,0]
    pc = getPixel_helper(depth,xs,ys,val,dataset,representation).T

    return pc,nn

def drawMatch(img0,img1,src,tgt,color='b'):
    if len(img0.shape)==2:
      img0=np.expand_dims(img0,2)
    if len(img1.shape)==2:
      img1=np.expand_dims(img1,2)
    h,w = img0.shape[0],img0.shape[1]
    img = np.zeros([2*h,w,3])
    img[:h,:,:] = img0
    img[h:,:,:] = img1
    n = len(src)
    if color == 'b':
        color=(255,0,0)
    else:
        color=(0,255,0)
    for i in range(n):
      cv2.circle(img, (int(src[i,0]), int(src[i,1])), 3,color,-1)
      cv2.circle(img, (int(tgt[i,0]), int(tgt[i,1])+h), 3,color,-1)
      cv2.line(img, (int(src[i,0]),int(src[i,1])),(int(tgt[i,0]),int(tgt[i,1])+h),color,1)
    return img

def getKeypoint(rs,rt,feats,featt):
    h,w=160,640
    grays= cv2.cvtColor(rs,cv2.COLOR_BGR2GRAY)
    grayt= cv2.cvtColor(rt,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.02)
    
    grays=grays[:,160:160*2]
    (kps, _) = sift.detectAndCompute(grays, None)
    if not len(kps):
        return None,None,None,None,None,None
    pts=np.zeros([len(kps),2])
    for j,m in enumerate(kps):
        pts[j,:] = m.pt
    pts[:,0]+=160

    grayt=grayt[:,160:160*2]
    (kpt, _) = sift.detectAndCompute(grayt, None)
    if not len(kpt):
        return None,None,None,None,None,None
    ptt=np.zeros([len(kpt),2])
    for j,m in enumerate(kpt):
        ptt[j,:] = m.pt
    ptt[:,0]+=160

    ptsNorm = pts.copy().astype('float')
    ptsNorm[:,0]/=640
    ptsNorm[:,1]/=160
    pttNorm = ptt.copy().astype('float')
    pttNorm[:,0]/=640
    pttNorm[:,1]/=160

    fs0 = interpolate(feats,torch_op.v(ptsNorm))
    ft0 = interpolate(featt,torch_op.v(pttNorm))

    # find the most probable correspondence using feature map
    C = feats.shape[0]
    fsselect=np.random.choice(range(pts.shape[0]),min(30,pts.shape[0]))
    ftselect=np.random.choice(range(ptt.shape[0]),min(30,ptt.shape[0]))
    dist=(fs0[:,fsselect].unsqueeze(2) - featt.view(C,1,-1)).pow(2).sum(0).view(len(fsselect),h,w)
    

    pttAug=Sampling(torch_op.npy(dist),2)
    dist=(ft0[:,ftselect].unsqueeze(2) - feats.view(C,1,-1)).pow(2).sum(0).view(len(ftselect),h,w)
    ptsAug=Sampling(torch_op.npy(dist),2)

    pttAug=pttAug.reshape(-1,2)
    ptsAug=ptsAug.reshape(-1,2)
    valid=(pttAug[:,0]<w-1)*(pttAug[:,1]<h-1)
    pttAug=pttAug[valid]
    valid=(ptsAug[:,0]<w-1)*(ptsAug[:,1]<h-1)
    ptsAug=ptsAug[valid]

    pts = np.concatenate((pts,ptsAug))
    ptt = np.concatenate((ptt,pttAug))

    N=30
    xs=(np.random.rand(N)*640).astype('int').clip(0,640-2)
    ys=(np.random.rand(N)*160).astype('int').clip(0,160-2)
    ptsrnd=np.stack((xs,ys),1)
    valid=((ptsrnd[:,0]>=160) *(ptsrnd[:,0]<=160*2))
    ptsrnd=ptsrnd[~valid]
    ptsrndNorm = ptsrnd.copy().astype('float')
    ptsrndNorm[:,0]/=640
    ptsrndNorm[:,1]/=160
    fs0 = interpolate(feats,torch_op.v(ptsrndNorm))
    fsselect=np.random.choice(range(ptsrnd.shape[0]),min(100,ptsrnd.shape[0]))
    dist=(fs0[:,fsselect].unsqueeze(2) - featt.view(C,1,-1)).pow(2).sum(0).view(len(fsselect),h,w)
    
    pttAug=Sampling(torch_op.npy(dist),2)
    pttAug=pttAug.reshape(-1,2)
    valid=(pttAug[:,0]<w-1)*(pttAug[:,1]<h-1)
    pttAug=pttAug[valid]
    pts=np.concatenate((pts,ptsrnd[fsselect]))
    ptt=np.concatenate((ptt,pttAug))

    ptsNorm = pts.copy().astype('float')
    ptsNorm[:,0]/=640
    ptsNorm[:,1]/=160

    pttNorm = ptt.copy().astype('float')
    pttNorm[:,0]/=640
    pttNorm[:,1]/=160

    valid=(pts[:,0]>=160) *(pts[:,0]<=160*2)
    ptsW = np.ones(len(valid))
    ptsW[~valid]*=0.99

    valid=(ptt[:,0]>=160) *(ptt[:,0]<=160*2)
    pttW = np.ones(len(valid))
    pttW[~valid]*=0.99

    return pts,ptsNorm,ptsW,ptt,pttNorm,pttW


def getKeypoint_kinect(rs,rt,feats,featt,rs_full,rt_full):
    h,w=160,640
    grays= cv2.cvtColor(rs,cv2.COLOR_BGR2GRAY)
    grayt= cv2.cvtColor(rt,cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.02)
    grays = cv2.cvtColor(rs_full,cv2.COLOR_BGR2GRAY)
    (kps, _) = sift.detectAndCompute(grays, None)
    if not len(kps):
        return None,None,None,None,None,None
    pts=np.zeros([len(kps),2])
    for j,m in enumerate(kps):
        pts[j,:] = m.pt
    pts[:,0] = pts[:,0]/640*88 # the observed region size of kinect camera is [88x66]
    pts[:,1] = pts[:,1]/480*66
    pts[:,0]+=160+80-44
    pts[:,1]+=80-33

    grayt = cv2.cvtColor(rt_full,cv2.COLOR_BGR2GRAY)
    (kpt, _) = sift.detectAndCompute(grayt, None)
    if not len(kpt):
        return None,None,None,None,None,None
    ptt=np.zeros([len(kpt),2])
    for j,m in enumerate(kpt):
        ptt[j,:] = m.pt
    ptt[:,0] = ptt[:,0]/640*88
    ptt[:,1] = ptt[:,1]/480*66
    ptt[:,0]+=160+80-44
    ptt[:,1]+=80-33

    pts=pts[np.random.choice(range(len(pts)),300),:]
    ptt=ptt[np.random.choice(range(len(ptt)),300),:]
    
    ptsNorm = pts.copy().astype('float')
    ptsNorm[:,0]/=640
    ptsNorm[:,1]/=160
    pttNorm = ptt.copy().astype('float')
    pttNorm[:,0]/=640
    pttNorm[:,1]/=160

    fs0 = interpolate(feats,torch_op.v(ptsNorm))
    ft0 = interpolate(featt,torch_op.v(pttNorm))

    # find the most probable correspondence using feature map
    C = feats.shape[0]
    fsselect=np.random.choice(range(pts.shape[0]),min(30,pts.shape[0]))
    ftselect=np.random.choice(range(ptt.shape[0]),min(30,ptt.shape[0]))
    dist=(fs0[:,fsselect].unsqueeze(2) - featt.view(C,1,-1)).pow(2).sum(0).view(len(fsselect),h,w)
    pttAug=Sampling(torch_op.npy(dist),2)
    dist=(ft0[:,ftselect].unsqueeze(2) - feats.view(C,1,-1)).pow(2).sum(0).view(len(ftselect),h,w)
    ptsAug=Sampling(torch_op.npy(dist),2)

    pttAug=pttAug.reshape(-1,2)
    ptsAug=ptsAug.reshape(-1,2)
    valid=(pttAug[:,0]<w-1)*(pttAug[:,1]<h-1)
    pttAug=pttAug[valid]
    valid=(ptsAug[:,0]<w-1)*(ptsAug[:,1]<h-1)
    ptsAug=ptsAug[valid]

    pts = np.concatenate((pts,ptsAug))
    ptt = np.concatenate((ptt,pttAug))

    N=120
    xs=(np.random.rand(N)*640).astype('int').clip(0,640-2)
    ys=(np.random.rand(N)*160).astype('int').clip(0,160-2)
    ptsrnd=np.stack((xs,ys),1)

    # filter out observed region
    valid=((ptsrnd[:,0]>=160+80-44) *(ptsrnd[:,0]<=160+80+44)*(ptsrnd[:,1]>=80-33) *(ptsrnd[:,1]<=80+33))
    ptsrnd=ptsrnd[~valid]

    ptsrndNorm = ptsrnd.copy().astype('float')
    ptsrndNorm[:,0]/=640
    ptsrndNorm[:,1]/=160
    fs0 = interpolate(feats,torch_op.v(ptsrndNorm))
    fsselect=np.random.choice(range(ptsrnd.shape[0]),min(100,ptsrnd.shape[0]))
    dist=(fs0[:,fsselect].unsqueeze(2) - featt.view(C,1,-1)).pow(2).sum(0).view(len(fsselect),h,w)

    pttAug=Sampling(torch_op.npy(dist),2)
    pttAug=pttAug.reshape(-1,2)
    valid=(pttAug[:,0]<w-1)*(pttAug[:,1]<h-1)
    pttAug=pttAug[valid]
    pts=np.concatenate((pts,ptsrnd[fsselect]))
    ptt=np.concatenate((ptt,pttAug))

    ptsNorm = pts.copy().astype('float')
    ptsNorm[:,0]/=640
    ptsNorm[:,1]/=160

    pttNorm = ptt.copy().astype('float')
    pttNorm[:,0]/=640
    pttNorm[:,1]/=160

    # hacks to get the observed region for kinect camera configuration. 
    valid=((pts[:,0]>=160+80-44) *(pts[:,0]<=160+80+44)*(pts[:,1]>=80-33) *(pts[:,1]<=80+33))
    ptsW = np.ones(len(valid))
    ptsW[~valid]*=0.99

    valid=((ptt[:,0]>=160+80-44) *(ptt[:,0]<=160+80+44)*(ptt[:,1]>=80-33) *(ptt[:,1]<=80+33))
    pttW = np.ones(len(valid))
    pttW[~valid]*=0.99

    return pts,ptsNorm,ptsW,ptt,pttNorm,pttW

def Sampling(heatmap,K):
    # heatmap: [n,h,w]
    # return: [n,K,2]
    heatmap = np.exp(-heatmap/2)
    n,h,w=heatmap.shape
    pt = np.zeros([n,K,2])
    wsz=15
    for i in range(n):
        for j in range(K):
            idx=np.argmax(heatmap[i])
            coord=np.unravel_index(idx,heatmap[i].shape)[::-1]
            pt[i,j,:]=coord
            # suppress the neighbors
            topl=[max(0,coord[0]-wsz),max(0,coord[1]-wsz)]
            botr=[min(w-1,coord[0]+wsz),min(h-1,coord[1]+wsz)]
            heatmap[i][topl[1]:botr[1],topl[0]:botr[0]] = heatmap[i].min()
    return pt
