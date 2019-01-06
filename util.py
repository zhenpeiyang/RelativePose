import numpy as np
from utils import train_op,torch_op
import io
import torch
import config
import sys
import collections
import cv2
import torch.nn.functional as F
from numpy.random import randn
from torch.autograd import Variable
import torch.nn as nn
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)
from sklearn.neighbors import KDTree

def point_cloud_overlap(pc_src,pc_tgt,R_gt_44):
    pc_src_trans = np.matmul(R_gt_44[:3,:3],pc_src.T) +R_gt_44[:3,3:4]
    tree = KDTree(pc_tgt)
    nearest_dist, nearest_ind = tree.query(pc_src_trans.T, k=1)
    nns2t = np.min(nearest_dist)
    hasCorres=(nearest_dist < 0.08)
    overlap_val_s2t = hasCorres.sum()/pc_src.shape[0]

    pc_tgt_trans = np.matmul(np.linalg.inv(R_gt_44),np.concatenate((pc_tgt.T,np.ones([1,pc_tgt.shape[0]]))))[:3,:]
    tree = KDTree(pc_src)
    nearest_dist, nearest_ind = tree.query(pc_tgt_trans.T, k=1)
    nnt2s = np.min(nearest_dist)
    hasCorres=(nearest_dist < 0.08)
    overlap_val_t2s = hasCorres.sum()/pc_tgt.shape[0]

    overlap_val = max(overlap_val_s2t,overlap_val_t2s)
    cam_dist_this = np.linalg.norm(R_gt_44[:3,3])
    pc_dist_this = np.linalg.norm(pc_src_trans.mean(1) - pc_tgt.T.mean(1))
    pc_nn = (nns2t+nnt2s)/2
    return overlap_val,cam_dist_this,pc_dist_this,pc_nn

def parse_data(depth,rgb,norm,dataList,method):
    if 'suncg' in dataList or 'matterport' in dataList:
        depth_src = depth[0,0,:,160:160*2]
        depth_tgt = depth[0,1,:,160:160*2]
        color_src = rgb[0,0,:,:,160:160*2].transpose(1,2,0)
        color_tgt = rgb[0,1,:,:,160:160*2].transpose(1,2,0)
        
        normal_src = norm[0,0,:,:,160:160*2].copy().transpose(1,2,0)
        normal_tgt = norm[0,1,:,:,160:160*2].copy().transpose(1,2,0)
        pc_src,mask_src = depth2pc(depth_src, dataList)
        pc_tgt,mask_tgt = depth2pc(depth_tgt, dataList)
        color_src = color_src.reshape(-1,3)[mask_src]/255.
        color_tgt = color_tgt.reshape(-1,3)[mask_tgt]/255.
        normal_src = normal_src.reshape(-1,3)[mask_src]
        normal_tgt = normal_tgt.reshape(-1,3)[mask_tgt]

    elif 'scannet' in dataList:
        if 'ours' in method:
            depth_src = depth[0,0,80-33:80+33,160+80-44:160+80+44]
            depth_tgt = depth[0,1,80-33:80+33,160+80-44:160+80+44]
            color_src = rgb[0,0,:,80-33:80+33,160+80-44:160+80+44].transpose(1,2,0)
            color_tgt = rgb[0,1,:,80-33:80+33,160+80-44:160+80+44].transpose(1,2,0)
            normal_src = norm[0,0,:,80-33:80+33,160+80-44:160+80+44].copy().transpose(1,2,0)
            normal_tgt = norm[0,1,:,80-33:80+33,160+80-44:160+80+44].copy().transpose(1,2,0)

            pc_src,mask_src = depth2pc(depth_src, dataList)
            pc_tgt,mask_tgt = depth2pc(depth_tgt, dataList)
            color_src = color_src.reshape(-1,3)[mask_src]/255.
            color_tgt = color_tgt.reshape(-1,3)[mask_tgt]/255.
            normal_src = normal_src.reshape(-1,3)[mask_src]
            normal_tgt = normal_tgt.reshape(-1,3)[mask_tgt]
            normal_src=normal_src/np.linalg.norm(normal_src,axis=1,keepdims=True)
            normal_tgt=normal_tgt/np.linalg.norm(normal_tgt,axis=1,keepdims=True)
            where_are_NaNs = np.isnan(normal_src.sum(1))
            normal_src[where_are_NaNs] = 0
            where_are_NaNs = np.isnan(normal_tgt.sum(1))
            normal_tgt[where_are_NaNs] = 0

        else:
            depth_src = depth[0,0,:,:]
            depth_tgt = depth[0,1,:,:]
            color_src = rgb[0,0,:,:].transpose(1,2,0)
            color_tgt = rgb[0,1,:,:].transpose(1,2,0)
            pc_src,mask_src = depth2pc(depth_src, dataList)
            pc_tgt,mask_tgt = depth2pc(depth_tgt, dataList)
            color_src = color_src.reshape(-1,3)[mask_src]/255.
            color_tgt = color_tgt.reshape(-1,3)[mask_tgt]/255.
            normal_src=None
            normal_tgt=None

    return depth_src,depth_tgt,normal_src,normal_tgt,color_src,color_tgt,pc_src,pc_tgt

def warping(view,R,dataList):
    if np.linalg.norm(R-np.eye(4)) == 0:
        return torch.zeros(view.shape).float().cuda()
    if 'suncg' in dataList:
        h=160
        colorpct=[]
        normalpct=[]
        depthpct=[]
        rgb=view[0,0:3,:,:].transpose(1,2,0)
        normal=view[0,3:6,:,:].transpose(1,2,0)
        depth=view[0,6,:,:]
        
        for ii in range(4):
            colorpct.append(rgb[:,ii*h:(ii+1)*h,:].reshape(-1,3))
            normalpct.append(normal[:,ii*h:(ii+1)*h,:].reshape(-1,3))
            depthpct.append(depth[:,ii*h:(ii+1)*h].reshape(-1))
        colorpct=np.concatenate(colorpct)
        normalpct=np.concatenate(normalpct)
        depthpct=np.concatenate(depthpct)
        # get the coordinates of each point in the first coordinate system
        pct = Pano2PointCloud(depth,'suncg')# be aware of the order of returned pc!!!
        R_this_p=R
        pct_reproj = np.matmul(R_this_p,np.concatenate((pct,np.ones([1,pct.shape[1]]))))[:3,:]

        # assume always observe the second view(right view)
        colorpct=colorpct[h*h:h*h*2,:]
        depthpct=depthpct[h*h:h*h*2]
        normalpct=normalpct[h*h:h*h*2,:]
        normalpct=np.matmul(R_this_p[:3,:3], normalpct.T).T
        pct_reproj=pct_reproj[:,h*h:h*h*2]
        
        s2t_rgb_p=reproj_helper(pct_reproj,colorpct,rgb.shape,'color',dataList)
        s2t_n_p=reproj_helper(pct_reproj,normalpct,rgb.shape,'normal',dataList)
        s2t_d_p=reproj_helper(pct_reproj,depthpct,rgb.shape[:2],'depth',dataList)
        
        s2t_mask_p=(s2t_d_p!=0).astype('int')
        view_s2t=np.expand_dims(np.concatenate((s2t_rgb_p,s2t_n_p,np.expand_dims(s2t_d_p,2),np.expand_dims(s2t_mask_p,2)),2),0).transpose(0,3,1,2)
    elif 'matterport' in dataList:
        rgb=view[0,0:3,:,:].transpose(1,2,0)
        normal=view[0,3:6,:,:].transpose(1,2,0)
        depth=view[0,6,:,:]
        h=160
        pct,mask = depth2pc(depth[:,160:160*2],'matterport')# be aware of the order of returned pc!!!
        ii=1
        colorpct=rgb[:,ii*h:(ii+1)*h,:].reshape(-1,3)[mask,:]
        normalpct=normal[:,ii*h:(ii+1)*h,:].reshape(-1,3)[mask,:]
        depthpct=depth[:,ii*h:(ii+1)*h].reshape(-1)[mask]
        # get the coordinates of each point in the first coordinate system
        R_this_p = R
        pct_reproj = np.matmul(R_this_p,np.concatenate((pct.T,np.ones([1,pct.shape[0]]))))[:3,:]
        normalpct=np.matmul(R_this_p[:3,:3], normalpct.T).T
        s2t_rgb_p=reproj_helper(pct_reproj,colorpct,rgb.shape,'color',dataList)
        s2t_n_p=reproj_helper(pct_reproj,normalpct,rgb.shape,'normal',dataList)
        s2t_d_p=reproj_helper(pct_reproj,depthpct,rgb.shape[:2],'depth',dataList)
        s2t_mask_p=(s2t_d_p!=0).astype('int')
        view_s2t=np.expand_dims(np.concatenate((s2t_rgb_p,s2t_n_p,np.expand_dims(s2t_d_p,2),np.expand_dims(s2t_mask_p,2)),2),0).transpose(0,3,1,2)

    elif 'scannet' in dataList:
        assert(view.shape[2]==160 and view.shape[3]==640)
        h=view.shape[2]
        rgb=view[0,0:3,:,:].transpose(1,2,0)
        normal=view[0,3:6,:,:].transpose(1,2,0)
        depth=view[0,6,:,:]

        pct,mask = depth2pc(depth[80-33:80+33,160+80-44:160+80+44],'scannet')# be aware of the order of returned pc!!!
        colorpct = rgb[80-33:80+33,160+80-44:160+80+44,:].reshape(-1,3)[mask]
        normalpct = normal[80-33:80+33,160+80-44:160+80+44,:].reshape(-1,3)[mask]
        depthpct = depth[80-33:80+33,160+80-44:160+80+44].reshape(-1)[mask]

        R_this_p=R
        pct_reproj = np.matmul(R_this_p,np.concatenate((pct.T,np.ones([1,pct.shape[0]]))))[:3,:]
        normalpct=np.matmul(R_this_p[:3,:3], normalpct.T).T
        s2t_rgb_p=reproj_helper(pct_reproj,colorpct,rgb.shape,'color',dataList)
        s2t_n_p=reproj_helper(pct_reproj,normalpct,rgb.shape,'normal',dataList)
        s2t_d_p=reproj_helper(pct_reproj,depthpct,rgb.shape[:2],'depth',dataList)
        s2t_mask_p=(s2t_d_p!=0).astype('int')

        view_s2t=np.expand_dims(np.concatenate((s2t_rgb_p,s2t_n_p,np.expand_dims(s2t_d_p,2),np.expand_dims(s2t_mask_p,2)),2),0).transpose(0,3,1,2)
    return view_s2t



def angular_distance_np(R_hat, R):
    # measure the angular distance between two rotation matrice
    # R1,R2: [n, 3, 3]
    if R_hat.shape == (3,3):
        R_hat = R_hat[np.newaxis,:]
    if R.shape == (3,3):
        R = R[np.newaxis,:]
    n = R.shape[0]
    trace_idx = [0,4,8]
    trace = np.matmul(R_hat, R.transpose(0,2,1)).reshape(n,-1)[:,trace_idx].sum(1)
    metric = np.arccos(((trace - 1)/2).clip(-1,1)) / np.pi * 180.0
    return metric

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def read_super4pcs_mat(path):
    with open(path, 'r') as f:
        lines = f.readlines()[2:]
        r = np.zeros([4,4])
        for i in range(4):
            line = lines[i]
            tmp = line.strip('\n ').split('  ')
            tmp = [float(v) for v in tmp]
            r[i,:] = tmp
        return r

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
        dw = int(89.67//2) # 44
        dh = int(67.25//2) # 33
        tp[:,:,80-dh:80+dh,160+80-dw:160+80+dw]=1
        geow = 1-torch_op.v(tp)
    tp=torch_op.v(tp)
    x=x*tp
    return x,tp,geow

def randomRotation(epsilon):
    axis=(np.random.rand(3)-0.5)
    axis/=np.linalg.norm(axis)
    dtheta=np.random.randn(1)*np.pi*epsilon
    K=np.array([0,-axis[2],axis[1],axis[2],0,-axis[0],-axis[1],axis[0],0]).reshape(3,3)
    dR=np.eye(3)+np.sin(dtheta)*K+(1-np.cos(dtheta))*np.matmul(K,K)
    return dR

def horn87_v1(src, tgt, weight=None): # does not substract center, compare to horn87_np_v2
    '''
    # src: [(k), 3, n]
    # tgt: [(k), 3, n]
    # weight: [(k), n]
    # return: 
    #   (R, t) ([(k),3,3], [(k),3,1])
    '''
    if len(src.shape) == 2 and len(tgt.shape) == 2:
        src, tgt = src.unsqueeze(0), tgt.unsqueeze(0)
    assert(src.shape[2] == tgt.shape[2])
    nPts = src.shape[2]
    k = src.shape[0]
    has_weight=False
    if weight is None:
        weight = torch.ones(k,1,nPts).cuda().float()
    else:
        has_weight=True
        weight = weight.view(k,1,nPts)
    weight = weight / weight.sum(2,keepdim=True)
    src_ = src
    tgt_ = tgt
    if has_weight:
        for i in range(k):
            tgt_[i] *= weight[i]

    H = torch.bmm(src_, tgt_.transpose(2,1))
    R_ret = []
    for i in range(k):
        try:
            u, s, v = torch.svd(H[i,:,:].cpu())
            R = torch.matmul(v, u.t())
            det = torch.det(R)
            if det < 0:
                R = torch.matmul(v, torch.matmul(torch.diagflat(torch.FloatTensor([1,1,-1])),u.t()))
            R_ret.append(R.view(-1,3,3))

        except:
            print('rigid transform failed to converge')
            print('H:{}'.format(torch_op.npy(H)))

            R_ret.append(Variable(torch.eye(3).view(1,3,3), requires_grad=True))
    
    R_ret = torch.cat(R_ret).cuda()

    return R_ret

def horn87_np_v2(src, tgt,weight=None):
    '''
    # src: [(k), 3, n]
    # tgt: [(k), 3, n]
    # weight: [(k), n]
    # return: 
    #   (R, t) ([(k),3,3], [(k),3,1])
    '''
    if len(src.shape) == 2 and len(tgt.shape) == 2:
        src, tgt = src[np.newaxis,:], tgt[np.newaxis,:]
    assert(src.shape[2] == tgt.shape[2])
    nPts = src.shape[2]
    k = src.shape[0]
    has_weight=False
    if weight is None:
        weight = np.ones([k,1,nPts])
    else:
        has_weight=True
        weight = weight.reshape(k,1,nPts)

    src_ = src
    tgt_ = tgt

    if has_weight:
        tgt_ = tgt_.copy()
        for i in range(k):
            tgt_[i] *= weight[i]
    M = np.matmul(src_, tgt_.transpose(0,2,1))
    R_ret = []
    for i in range(k):
        N = np.array([[M[i,0, 0] + M[i,1, 1] + M[i,2, 2], M[i,1, 2] - M[i,2, 1], M[i,2, 0] - M[i,0, 2], M[i,0, 1] - M[i,1, 0]], 
                        [M[i,1, 2] - M[i,2, 1], M[i,0, 0] - M[i,1, 1] - M[i,2, 2], M[i,0, 1] + M[i,1, 0], M[i,0, 2] + M[i,2, 0]], 
                        [M[i,2, 0] - M[i,0, 2], M[i,0, 1] + M[i,1, 0], M[i,1, 1] - M[i,0, 0] - M[i,2, 2], M[i,1, 2] + M[i,2, 1]], 
                        [M[i,0, 1] - M[i,1, 0], M[i,2, 0] + M[i,0, 2], M[i,1, 2] + M[i,2, 1], M[i,2, 2] - M[i,0, 0] - M[i,1, 1]]])
        v, u = np.linalg.eig(N)
        id = v.argmax()

        q = u[:, id]
        R_ret.append(np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])], 
                        [2*(q[2]*q[1]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])], 
                        [2*(q[3]*q[1]-q[0]*q[2]), 2*(q[3]*q[2]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]]).reshape(1,3,3))
    R_ret = np.concatenate(R_ret)
    return R_ret

def drawMatch(img0,img1,src,tgt,color=['b']):
    if len(img0.shape)==2:
      img0=np.expand_dims(img0,2)
    if len(img1.shape)==2:
      img1=np.expand_dims(img1,2)
    h,w = img0.shape[0],img0.shape[1]
    img = np.zeros([2*h,w,3])
    img[:h,:,:] = img0
    img[h:,:,:] = img1
    n = len(src)
    colors=[]
    if len(color)!=1:
        assert(len(color) == n)
        for i in range(n):
            if color[i] == 'b':
                colors.append((255,0,0))
            elif color[i] == 'r':
                colors.append((0,0,255))
    else:
        for i in range(n):
            if color[0] == 'b':
                colors.append((255,0,0))
            elif color[0] == 'r':
                colors.append((0,0,255))
    for i in range(n):
      cv2.circle(img, (int(src[i,0]), int(src[i,1])), 3,colors[i],-1)
      cv2.circle(img, (int(tgt[i,0]), int(tgt[i,1])+h), 3,colors[i],-1)
      cv2.line(img, (int(src[i,0]),int(src[i,1])),(int(tgt[i,0]),int(tgt[i,1])+h),colors[i],3)
    return img

def drawKeypoint(imgSize, pts):
    # imgSize: [h,w]
    # pts: [n,2]
    ret=np.zeros(imgSize)
    color=(255,0,0)
    for i in range(len(pts)):
        cv2.circle(ret,(int(pts[i,0]), int(pts[i,1])), 3,color,-1)
    return ret

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]

def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = cos(theta)
    x = x * sin(theta)
    y = y * sin(theta)
    z = z * sin(theta)
    return w, x, y, z

def q_to_axisangle(q):
    w, v = q[0], q[1:]
    theta = acos(w) * 2.0
    return normalize(v), theta

def rot2Quaternion(rot):
    # rot: [3,3]
    assert(rot.shape==(3,3))
    tr = np.trace(rot)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qz = (rot[2,1]-rot[1,2]) / S
        qy = (rot[0,2]-rot[2,0]) / S
        qx = (rot[1,0]-rot[0,1]) / S
    elif (rot[0,0]>rot[1,1]) and (rot[0,0]>rot[2,2]):
        S = np.sqrt(1.0 + rot[0,0] - rot[1,1] - rot[2,2]) * 2
        qw = (rot[2,1] - rot[1,2]) / S
        qz = 0.25 * S
        qy = (rot[0,1]+rot[1,0]) / S
        qx = (rot[0,2]+rot[2,0]) / S
    elif rot[1,1] > rot[2,2]:
        S = np.sqrt(1.0 + rot[1,1] - rot[0,0] - rot[2,2]) * 2
        qw = (rot[0,2] - rot[2,0]) / S
        qz = (rot[0,1] + rot[1,0]) / S
        qy = 0.25 * S
        qx = (rot[1,2]+rot[2,1]) / S
    else:
        S = np.sqrt(1.0 + rot[2,2] - rot[0,0] - rot[1,1]) * 2
        qw = (rot[1,0] - rot[0,1]) / S
        qz = (rot[0,2] + rot[2,0]) / S
        qy = (rot[1,2] + rot[2,1]) / S
        qx = 0.25 * S

    return np.array([qw, qz, qy, qx])

def quaternion2Rot(q):
    # q:[4]
    # R:[3,3]
    R = np.zeros([3,3])
    R[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    R[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[0,2] = 2*(q[0]*q[2] + q[1]*q[3])
    R[1,0] = 2*(q[1]*q[2] + q[0]*q[3])
    R[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    R[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
    R[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
    R[2,1] = 2*(q[0]*q[1]+q[2]*q[3])
    R[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    return R

def Rnd(x):
  return max(-2 * x, min(2 * x, randn() * x))
  
def Flip(img):
  if len(img.shape) == 3:
    return img[:, :, ::-1].copy()  
  elif len(img.shape) == 4:
    return img[:, :, :, ::-1].copy()  
  else:
    raise Exception('Flip shape error')

def depth2pc(depth,dataList):
    
    if 'suncg' in dataList:
        w,h = depth.shape[1], depth.shape[0]
        assert(w == 160 and h == 160)
        Rs = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        # transform from ith frame to 0th frame
        ys, xs = np.meshgrid(range(h),range(h),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / h-0.5)*2
        zs = depth.flatten()
        mask = (zs!=0)
        zs = zs[mask]
        xs=xs.flatten()[mask]*zs
        ys=ys.flatten()[mask]*zs
        pc = np.stack((xs,ys,-zs),1)
        # assume second view!
        pc=np.matmul(Rs[:3,:3],pc.T).T
    elif 'matterport' in dataList:
        w,h = depth.shape[1], depth.shape[0]
        assert(w == 160 and h == 160)
        Rs = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        # transform from ith frame to 0th frame
        ys, xs = np.meshgrid(range(h),range(h),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / h-0.5)*2
        zs = depth.flatten()
        mask = (zs!=0)
        zs = zs[mask]
        xs=xs.flatten()[mask]*zs
        ys=ys.flatten()[mask]*zs
        pc = np.stack((xs,ys,-zs),1)

    elif 'scannet' in dataList:
        if (depth.shape[0] == 480 and depth.shape[1] == 640):
            w,h = depth.shape[1], depth.shape[0]
            # transform from ith frame to 0th frame
            ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
            ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
            zs = depth.flatten()
            mask = (zs!=0)
            zs = zs[mask]
            xs=xs.flatten()[mask]*zs/(0.8921875*2)
            ys=ys.flatten()[mask]*zs/(1.1895*2)
            pc = np.stack((xs,ys,-zs),1)
        elif (depth.shape[0] == 66 and depth.shape[1] == 88):
            w,h = depth.shape[1], depth.shape[0]
            # transform from ith frame to 0th frame
            ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
            ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
            zs = depth.flatten()
            mask = (zs!=0)
            zs = zs[mask]
            xs=xs.flatten()[mask]*zs
            ys=ys.flatten()[mask]*zs
            pc = np.stack((xs*w/160,ys*h/160,-zs),1)

    return pc,mask

def PanoIdx(index,h,w):
    total=h*w
    single=total//4
    hidx = index//single
    rest=index % single
    ys,xs=np.unravel_index(rest, [h,h])
    xs += hidx*h
    idx = np.zeros([len(xs),2])
    idx[:,0]=xs
    idx[:,1]=ys
    return idx

def reproj_helper(pct,colorpct,out_shape,mode,dataList):
    if 'suncg' in dataList:
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        # find which plane they intersect with
        h=out_shape[0]
        tp=pct.copy()
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectf=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)
        if mode in ['color','normal']:
            colorf=colorpct[intersectf,:]
        elif mode == 'depth':
            colorf=-tp[2,intersectf]
        coordf=tp[:2,intersectf]
        coordf[0,:]=(coordf[0,:]+1)*0.5*h
        coordf[1,:]=(1-coordf[1,:])*0.5*h
        coordf=coordf.round().clip(0,h-1).astype('int')

        tp=np.matmul(Rs[1][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectr=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

        if mode in ['color','normal']:
            colorr=colorpct[intersectr,:]
        elif mode == 'depth':
            colorr=-tp[2,intersectr]

        coordr=tp[:2,intersectr]
        coordr[0,:]=(coordr[0,:]+1)*0.5*h
        coordr[1,:]=(1-coordr[1,:])*0.5*h
        coordr=coordr.round().clip(0,h-1).astype('int')
        coordr[0,:]+=h

        tp=np.matmul(Rs[2][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectb=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

        if mode in ['color','normal']:
            colorb=colorpct[intersectb,:]
        elif mode == 'depth':
            colorb=-tp[2,intersectb]

        coordb=tp[:2,intersectb]
        coordb[0,:]=(coordb[0,:]+1)*0.5*h
        coordb[1,:]=(1-coordb[1,:])*0.5*h
        coordb=coordb.round().clip(0,h-1).astype('int')
        coordb[0,:]+=h*2

        tp=np.matmul(Rs[3][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectl=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

        if mode in ['color','normal']:
            colorl=colorpct[intersectl,:]
        elif mode == 'depth':
            colorl=-tp[2,intersectl]

        coordl=tp[:2,intersectl]
        coordl[0,:]=(coordl[0,:]+1)*0.5*h
        coordl[1,:]=(1-coordl[1,:])*0.5*h
        coordl=coordl.round().clip(0,h-1).astype('int')
        coordl[0,:]+=h*3

        proj=np.zeros(out_shape)

        proj[coordf[1,:],coordf[0,:]]=colorf
        proj[coordl[1,:],coordl[0,:]]=colorl
        proj[coordb[1,:],coordb[0,:]]=colorb
        proj[coordr[1,:],coordr[0,:]]=colorr
    elif 'matterport' in dataList:
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        h=out_shape[0]
        tp=np.matmul(Rs[3][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectf=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)
        if mode in ['color','normal']:
            colorf=colorpct[intersectf,:]
        elif mode == 'depth':
            colorf=-tp[2,intersectf]
        coordf=tp[:2,intersectf]
        coordf[0,:]=(coordf[0,:]+1)*0.5*h
        coordf[1,:]=(1-coordf[1,:])*0.5*h
        coordf=coordf.round().clip(0,h-1).astype('int')

        tp=np.matmul(Rs[0][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectr=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

        if mode in ['color','normal']:
            colorr=colorpct[intersectr,:]
        elif mode == 'depth':
            colorr=-tp[2,intersectr]

        coordr=tp[:2,intersectr]
        coordr[0,:]=(coordr[0,:]+1)*0.5*h
        coordr[1,:]=(1-coordr[1,:])*0.5*h
        coordr=coordr.round().clip(0,h-1).astype('int')
        coordr[0,:]+=h

        tp=np.matmul(Rs[1][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectb=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

        if mode in ['color','normal']:
            colorb=colorpct[intersectb,:]
        elif mode == 'depth':
            colorb=-tp[2,intersectb]

        coordb=tp[:2,intersectb]
        coordb[0,:]=(coordb[0,:]+1)*0.5*h
        coordb[1,:]=(1-coordb[1,:])*0.5*h
        coordb=coordb.round().clip(0,h-1).astype('int')
        coordb[0,:]+=h*2

        tp=np.matmul(Rs[2][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectl=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

        if mode in ['color','normal']:
            colorl=colorpct[intersectl,:]
        elif mode == 'depth':
            colorl=-tp[2,intersectl]

        coordl=tp[:2,intersectl]
        coordl[0,:]=(coordl[0,:]+1)*0.5*h
        coordl[1,:]=(1-coordl[1,:])*0.5*h
        coordl=coordl.round().clip(0,h-1).astype('int')
        coordl[0,:]+=h*3

        proj=np.zeros(out_shape)

        proj[coordf[1,:],coordf[0,:]]=colorf
        proj[coordl[1,:],coordl[0,:]]=colorl
        proj[coordb[1,:],coordb[0,:]]=colorb
        proj[coordr[1,:],coordr[0,:]]=colorr
    elif 'scannet' in dataList:
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        h=out_shape[0]
        tp=np.matmul(Rs[3][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectf=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)
        if mode in ['color','normal']:
            colorf=colorpct[intersectf,:]
        elif mode == 'depth':
            colorf=-tp[2,intersectf]
        coordf=tp[:2,intersectf]
        coordf[0,:]=(coordf[0,:]+1)*0.5*h
        coordf[1,:]=(1-coordf[1,:])*0.5*h
        coordf=coordf.round().clip(0,h-1).astype('int')

        tp=np.matmul(Rs[0][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectr=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

        if mode in ['color','normal']:
            colorr=colorpct[intersectr,:]
        elif mode == 'depth':
            colorr=-tp[2,intersectr]

        coordr=tp[:2,intersectr]
        coordr[0,:]=(coordr[0,:]+1)*0.5*h
        coordr[1,:]=(1-coordr[1,:])*0.5*h
        coordr=coordr.round().clip(0,h-1).astype('int')
        coordr[0,:]+=h

        tp=np.matmul(Rs[1][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectb=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

        if mode in ['color','normal']:
            colorb=colorpct[intersectb,:]
        elif mode == 'depth':
            colorb=-tp[2,intersectb]

        coordb=tp[:2,intersectb]
        coordb[0,:]=(coordb[0,:]+1)*0.5*h
        coordb[1,:]=(1-coordb[1,:])*0.5*h
        coordb=coordb.round().clip(0,h-1).astype('int')
        coordb[0,:]+=h*2

        tp=np.matmul(Rs[2][:3,:3].T,pct)
        tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
        intersectl=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

        if mode in ['color','normal']:
            colorl=colorpct[intersectl,:]
        elif mode == 'depth':
            colorl=-tp[2,intersectl]

        coordl=tp[:2,intersectl]
        coordl[0,:]=(coordl[0,:]+1)*0.5*h
        coordl[1,:]=(1-coordl[1,:])*0.5*h
        coordl=coordl.round().clip(0,h-1).astype('int')
        coordl[0,:]+=h*3

        proj=np.zeros(out_shape)

        proj[coordf[1,:],coordf[0,:]]=colorf
        proj[coordl[1,:],coordl[0,:]]=colorl
        proj[coordb[1,:],coordb[0,:]]=colorb
        proj[coordr[1,:],coordr[0,:]]=colorr
    return proj
    
def Pano2PointCloud(depth,dataList):
    # The order of rendered 4 view are different between suncg and scannet/matterport. 
    # Hacks to separately deal with each dataset and get the corrected assembled point cloud.
    # TODO: FIX THE DATASET INCONSISTENCY
    if 'suncg' in dataList:
        assert(depth.shape[0]==160 and depth.shape[1]==640)
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        w,h = depth.shape[1]//4, depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        pc = []
        for i in range(4):
            zs = depth[:,i*w:(i+1)*w].flatten()
            ys_this, xs_this = ys.flatten()*zs, xs.flatten()*zs
            pc_this = np.concatenate((xs_this,ys_this,-zs)).reshape(3,-1) # assume depth clean
            pc_this = np.matmul(Rs[i][:3,:3],pc_this)
            pc.append(pc_this)
        pc = np.concatenate(pc,1)
    elif 'matterport' in dataList:
        assert(depth.shape[0]==160 and depth.shape[1]==640)
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        w,h = depth.shape[1]//4, depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        pc = []
        for i in range(4):
            zs = depth[:,i*w:(i+1)*w].flatten()
            ys_this, xs_this = ys.flatten()*zs, xs.flatten()*zs
            pc_this = np.concatenate((xs_this,ys_this,-zs)).reshape(3,-1) # assume depth clean
            pc_this = np.matmul(Rs[(i-1)%4][:3,:3],pc_this)
            pc.append(pc_this)
        pc = np.concatenate(pc,1)
    elif 'scannet' in dataList:
        assert(depth.shape[0]==160 and depth.shape[1]==640)
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        w,h = depth.shape[1]//4, depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        pc = []
        for i in range(4):
            zs = depth[:,i*w:(i+1)*w].flatten()
            mask=(zs!=0)
            zs=zs[mask]
            ys_this, xs_this = ys.flatten()[mask]*zs/(1.1895*2), xs.flatten()[mask]*zs/(0.8921875*2)
            pc_this = np.concatenate((xs_this,ys_this,-zs)).reshape(3,-1) # assume depth clean
            pc_this = np.matmul(Rs[(i-1)%4][:3,:3],pc_this)
            pc.append(pc_this)
        pc = np.concatenate(pc,1)
    return pc

def saveimg(kk,filename='test.png'):
    cv2.imwrite(filename,(kk-kk.min())/(kk.max()-kk.min())*255)

def pnlayer(depth,normal,plane,dataList,representation):
    # dp: [n,1,h,w]
    # n: [n,3,h,w]
    if 'suncg' in dataList or 'matterport' in dataList:
        n,h,w = depth.shape[0],depth.shape[2],depth.shape[3]
        assert(h==w//4)
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        Rs=torch_op.v(Rs)
        loss_pn=0
        for i in range(4):
            plane_this=plane[:,0,:,i*h:(i+1)*h].contiguous()
            depth_this=depth[:,0,:,i*h:(i+1)*h].contiguous()
            ys, xs = np.meshgrid(range(h),range(h),indexing='ij')
            ys, xs = (0.5-ys / h)*2, (xs / h-0.5)*2
            xs = xs.flatten()
            ys = ys.flatten()
            zs = plane_this.view(-1)
            mask = (zs!=0)
            masknpy = torch_op.npy(mask)
            normal_this=normal[:,:,:,i*h:(i+1)*h].permute(0,2,3,1).contiguous().view(-1,3)
            if 'suncg' in dataList:
                normal_this=torch.matmul(Rs[i][:3,:3].t(),normal_this.t()).t()
            elif 'matterport' in dataList:
                normal_this=torch.matmul(Rs[(i-1)%4][:3,:3].t(),normal_this.t()).t()
            ray = np.tile(np.stack((-xs[masknpy],-ys[masknpy],np.ones(len(xs))),1),[n,1])
            ray = torch_op.v(ray)
            pcPn=(zs/(ray*normal_this+1e-6).sum(1)).unsqueeze(1)*ray
            
            xs=torch_op.v(np.tile(xs,n))
            ys=torch_op.v(np.tile(ys,n))
            zs=depth_this.view(-1)

            xs=xs*zs
            ys=ys*zs
            pcD = torch.stack((xs,ys,-zs),1)
            loss_pn+=(pcD-pcPn).clamp(-5,5).abs().mean()
    elif 'scannet' in dataList:
        raise Exception("not implemented: scannet/skybox representation")

    return loss_pn

def COSINELoss(input, target):
    loss = (1-(input.view(-1,3)*target.view(-1,3)).sum(1)).pow(2).mean()
    return loss

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def collate_fn_cat(batch):
  "Puts each data field into a tensor with outer dimension batch size"
  if torch.is_tensor(batch[0]):
    out = None
    return torch.cat(batch, 0, out=out)
    # for rnn variable length input
    """
    elif type(batch[0]).__name__ == 'list':
        import ipdb;ipdb.set_trace()
        dim = 0
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[dim], batch))
        # pad according to max_len
        #batch = map(lambda (x, y):(pad_tensor(x, pad=max_len, dim=dim), y), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
    """
  elif type(batch[0]).__module__ == 'numpy':
    elem = batch[0]
    if type(elem).__name__ == 'ndarray':
      """
      seq_length = np.array([b.shape[0] for b in batch])
      if (seq_length != seq_length.mean()).sum() > 0: # need paddding
        import ipdb;ipdb.set_trace()
        max_len = max(seq_length)
        perm = np.argsort(seq_length)[::-1]
        batch = batch[perm]
        return torch.stack(map(lambda x: pad_tensor(x, max_len, 0), batch))
      """
      try:
        torch.cat([torch.from_numpy(b) for b in batch], 0)
      except:
        import ipdb;ipdb.set_trace()
      return torch.cat([torch.from_numpy(b) for b in batch], 0)
    if elem.shape == ():  # scalars
      py_type = float if elem.dtype.name.startswith('float') else int
      return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
  elif isinstance(batch[0], int):
    return torch.LongTensor(batch)
  elif isinstance(batch[0], float):
    return torch.DoubleTensor(batch)
  elif isinstance(batch[0], string_classes):
    return batch
  elif isinstance(batch[0], collections.Mapping):
    return {key: collate_fn_cat([d[key] for d in batch]) for key in batch[0]}
  elif isinstance(batch[0], collections.Sequence):
    transposed = zip(*batch)
    return [collate_fn_cat(samples) for samples in transposed]

  raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def Rz(psi):
    m = np.zeros([3, 3])
    m[0, 0] = np.cos(psi)
    m[0, 1] = -np.sin(psi)
    m[1, 0] = np.sin(psi)
    m[1, 1] = np.cos(psi)
    m[2, 2] = 1
    return m

def Ry(phi):
    m = np.zeros([3, 3])
    m[0, 0] = np.cos(phi)
    m[0, 2] = np.sin(phi)
    m[1, 1] = 1
    m[2, 0] = -np.sin(phi)
    m[2, 2] = np.cos(phi)
    return m

def Rx(theta):
    m = np.zeros([3, 3])
    m[0, 0] = 1
    m[1, 1] = np.cos(theta)
    m[1, 2] = -np.sin(theta)
    m[2, 1] = np.sin(theta)
    m[2, 2] = np.cos(theta)
    return m

def pc2obj(filepath,pc):
    nverts = pc.shape[1]
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in range(nverts):
            f.write("v %.4f %.4f %.4f\n" % (pc[0,v],pc[1,v],pc[2,v]))

